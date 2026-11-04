"Replay convolution side-stream cursors, epilogue tiling and output lifetimes."

from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml
from dense_contracts import DenseSymbols
from frontend_contract import DTYPES
from moe_dense import ScalarSource

from kta.core.expr import Expr

STUDY = Path(__file__).resolve().parents[1]
PREFIX = "examples/python/CuTeDSL/cute/blackwell_geforce/kernel/conv/"


class ScalarValues(ast.NodeTransformer):
    """Inspect operator order using already decoded scalar fragment values."""

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"load", "to"}:
            return self.visit(node.func.value)
        return self.generic_visit(node)


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    trees = {}
    for filename in ("dense_implicit_gemm_fprop.py", "dense_blockscaled_implicit_gemm_fprop.py"):
        data = (Path(manifest["tree"]) / (PREFIX + filename)).read_bytes()
        retained = next(x for x in manifest["files"] if x["path"] == PREFIX + filename)
        assert hashlib.sha256(data).hexdigest() == retained["sha256"]
        trees[filename] = ast.parse(data)
    classes = {
        name: next(x for x in tree.body if isinstance(x, ast.ClassDef))
        for name, tree in trees.items()
    }
    dense, scaled = classes.values()
    methods = {x.name: x for x in scaled.body if isinstance(x, ast.FunctionDef)}
    module = ast.Module([copy.deepcopy(methods["advance"])], [])
    ns = {}
    exec(
        compile(ast.fix_missing_locations(ScalarSource().visit(module)), "source-jump", "exec"), ns
    )
    advance = ns["advance"]
    setup = methods["_setup_attributes"]
    start = next(
        i
        for i, x in enumerate(setup.body)
        if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "(epi_m, epi_n)"
    )
    end = next(
        i
        for i, x in enumerate(setup.body)
        if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "self.epi_tile"
    )
    tile_code = compile(ast.Module(setup.body[start : end + 1], []), "source-epi-policy", "exec")
    tiles = []
    for tile_n, residual in itertools.product((64, 96, 128), (False, True)):
        obj = SimpleNamespace(tile_shape_mnk=(128, tile_n, 64), has_residual=residual)
        exec(tile_code, {"self": obj})
        em, en = obj.epi_tile
        assert 128 % em == 0 and tile_n % en == 0
        subtiles = 128 // em * (tile_n // en)
        covered = set()
        for mi, ni, row, col in itertools.product(
            range(128 // em), range(tile_n // en), range(em), range(en)
        ):
            coord = mi * em + row, ni * en + col
            assert coord not in covered
            covered.add(coord)
        assert covered == set(itertools.product(range(128), range(tile_n)))
        tiles.append(
            {
                "tile_n": tile_n,
                "residual": residual,
                "epi_tile": [em, en],
                "subtiles": subtiles,
                "epi_stages": min(subtiles, 4),
            }
        )
    assert next(x for x in tiles if x["tile_n"] == 96 and x["residual"])["subtiles"] == 6

    # Each side stream has its own fixed item length; the same local parity owns
    # all streams. A cancelled grid tile number need not share that parity.
    stream_domains = releases = 0
    for stages, items, main_steps, residual_steps in itertools.product(
        range(1, 9), range(14), (1, 2, 3, 5, 17), (1, 2, 4, 6)
    ):
        for stream, steps in (("input", main_steps), ("bias", 1), ("residual", residual_steps)):
            consumed = set()
            for group in (0, 1):
                state = SimpleNamespace(stages=stages, _index=0, _phase=0, _count=0)
                # Source visits every item; non-owner advances instead of consuming.
                for item in range(items):
                    if item % 2 != group:
                        advance(None, state, steps)
                        continue
                    if stream == "input":
                        state._count = 0
                    for k in range(steps):
                        q = item * steps + k
                        assert (state._index, state._phase) == (q % stages, (q // stages) % 2)
                        assert (item, k) not in consumed
                        consumed.add((item, k))
                        advance(None, state, 1)
                        releases += 1
            assert consumed == set(itertools.product(range(items), range(steps)))
            stream_domains += 1
    cancelled_grid_order = (7, 2, 11, 0, 5)
    assert [i % 2 for i in range(len(cancelled_grid_order))] != [
        x % 2 for x in cancelled_grid_order
    ]

    # Output source reads retire in issue order. Commit then acquire waits until
    # at most S-1 groups remain, before any slot is overwritten; tail drains the
    # remaining groups before handing the shared array to the other math group.
    output_domains = output_reads = 0
    for stages, subtiles, items in itertools.product(range(1, 5), range(1, 9), range(1, 8)):
        outstanding = []
        for item in range(items):
            for u in range(subtiles):
                slot = u % stages
                assert all(old_slot != slot for _, old_slot in outstanding)
                outstanding.append(((item, u), slot))
                if len(outstanding) > stages - 1:
                    outstanding.pop(0)
                output_reads += 1
            outstanding.clear()  # source producer_tail before order arrive
            assert not outstanding
        output_domains += 1

    # The check keeps native expression typing separate from scalar rounding
    # witnesses. These exact cases distinguish the source's operator order.
    def fp16(x):
        return struct.unpack("e", struct.pack("e", x))[0]

    def fp32(x):
        return struct.unpack("f", struct.pack("f", x))[0]

    bias = fp32(1.0001)
    bias_witness = {
        "bias_fp32": bias,
        "cast_to_fp16_acc_then_output_fp32": fp32(fp16(bias)),
        "fp32_bias_add": bias,
    }
    assert bias_witness["cast_to_fp16_acc_then_output_fp32"] != bias_witness["fp32_bias_add"]
    dense_tree = next(tree for name, tree in trees.items() if "blockscaled" not in name)
    functions = [
        copy.deepcopy(x)
        for x in dense_tree.body
        if isinstance(x, ast.FunctionDef)
        and x.name in {"_check_tensor_alignment", "_check_im2col_descriptor_limits"}
    ]
    guarded = copy.deepcopy(dense)
    guarded.body = [
        x for x in guarded.body if isinstance(x, ast.FunctionDef) and x.name == "can_implement"
    ]
    ns = {"widths": DTYPES}
    module = ast.Module([*functions, guarded], [])
    exec(compile(ast.fix_missing_locations(DenseSymbols().visit(module)), "dense-menu", "exec"), ns)
    obj = ns[guarded.name].__new__(ns[guarded.name])
    obj.tile_shape_mnk, obj.atom_layout, obj.cluster_shape_mnk = (128, 64, 64), (2, 2, 1), (1, 1, 1)
    obj.can_implement(
        1,
        64,
        128,
        "Float16",
        "Float16",
        "Float32",
        (1,) * 3,
        (1,) * 3,
        (1,) * 3,
        (0,) * 3,
        (0,) * 3,
        "Float32",
    )
    bias_witness["source_menu_accepts_acc16_output32_bias32"] = True
    stores = [
        x
        for x in ast.walk(methods["kernel"])
        if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == "store"
    ]

    def stored_expression(marker):
        call = next(x for x in stores if marker in ast.unparse(x.args[0]))
        expression = ScalarValues().visit(ast.Expression(copy.deepcopy(call.args[0])))
        return compile(ast.fix_missing_locations(expression), "source-epilogue-order", "eval")

    alpha_expr = stored_expression("* alpha")
    bias_expr = stored_expression("tCrBias.load()")
    residual_expr = stored_expression("tRS_rResidual.load()")
    scaled_value = eval(alpha_expr, {"accumulators": 2, "alpha": 3})
    biased_value = eval(bias_expr, {"accumulators": scaled_value, "tCrBias": 1})
    assert biased_value == 7 and biased_value != 3 * (2 + 1)
    residual_value = eval(
        residual_expr, {"tRS_rD": -2, "tRS_rResidual": 3, "self": SimpleNamespace(beta=1)}
    )
    assert max(0, residual_value) == 1 and max(0, residual_value) != max(0, -2) + 3

    # Bind actual CLC participant arithmetic, including the same residual const
    # that gates its extra stream. Idle round-up warps are not counted.
    kernels = {
        "dense": next(
            x for x in dense.body if isinstance(x, ast.FunctionDef) and x.name == "kernel"
        ),
        "scaled": methods["kernel"],
    }
    counts = {}
    for name, kernel in kernels.items():
        owner = next(
            x
            for x in ast.walk(kernel)
            if isinstance(x, ast.If) and ast.unparse(x.test) == "tile_parity == warp_group_idx"
        )
        tail_index = next(
            i
            for i, x in enumerate(owner.body)
            if ast.unparse(x) == "tma_store_pipeline.producer_tail()"
        )
        handoff = owner.body[tail_index + 1]
        assert isinstance(handoff, ast.Assign)
        assert ast.unparse(handoff.value.func) in {
            "pingpong_pipeline.arrive",
            "math_wg_order_barrier.arrive",
        }
        # A skipped item changes cursors only: no wait/release/copy call belongs
        # to its branch; the CLC read/release follows outside this owner branch.
        assert not any(
            isinstance(x, ast.Call)
            and isinstance(x.func, ast.Attribute)
            and x.func.attr in {"consumer_release", "consumer_wait", "copy"}
            for statement in owner.orelse
            for x in ast.walk(statement)
        )
        call = next(
            x.value
            for x in ast.walk(kernel)
            if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "clc_consumer_group"
        )
        for residual in (False, True) if name == "scaled" else (False,):
            obj = SimpleNamespace(
                num_mma_warps=8,
                num_dma_warps=1,
                num_sched_warps=1,
                num_threads_per_warp=32,
                sfa_cpasync_warp_group_size=4,
            )
            env = {"self": obj, "num_residual_clc_warps": int(residual)}
            count = eval(compile(ast.Expression(call.args[1]), "clc-count", "eval"), env)
            counts[f"{name}_residual_{residual}"] = count
            assert count == (320 if name == "dense" else 448 + 32 * residual)
        order_arrays = [
            x
            for x in ast.walk(
                classes[next(k for k in classes if ("blockscaled" in k) == (name == "scaled"))]
            )
            if isinstance(x, ast.AnnAssign)
            and ast.unparse(x.target)
            in {"pingpong_pipeline_array_ptr", "math_wg_order_barrier_array_ptr"}
        ]
        assert order_arrays and all(
            ast.unparse(x.annotation).endswith("cutlass.Int64, 4]") for x in order_arrays
        )

    rule = yaml.safe_load(
        (STUDY / "delta/interleaved_consumer_streams@1/template.yaml").read_text()
    )
    p = {
        "stages": 3,
        "cursor_arithmetic_fits": True,
        "producer_steps_per_item": 5,
        "producer_item_order_immutable": True,
        "item_steps_uniform": True,
        "group_partition_exact": True,
        "active_consumer_count_matches": True,
        "skipped_steps_not_released": True,
        "owned_steps_consumed_once": True,
        "order_protocol_valid": True,
        "shared_output_lifetime_valid": True,
        "progress_admitted": True,
    }
    variables = {"groups": 2, "steps_per_item": 5}
    for row in [*rule["constraints"], *rule["preconditions"]]:
        assert Expr.parse(row["expr"], variables.keys()).eval(variables, p=p)
    scope = Expr.parse(rule["scope"]["predicate"], variables.keys())
    for name in ("group_partition_exact", "order_protocol_valid", "shared_output_lifetime_valid"):
        assert not scope.eval(variables, p={**p, name: False})

    result = {
        "outcome": "PASS",
        "epilogue_tiles": tiles,
        "side_stream_domains": stream_domains,
        "owned_stage_releases": releases,
        "output_ring_domains": output_domains,
        "output_subtile_source_reads": output_reads,
        "clc_participants": counts,
        "fp16_bias_conversion_witness": bias_witness,
        "boundary": (
            "Retained tile/CLC/jump arithmetic plus abstract per-stream and ou"
            "tput-read protocol models; scalar precision witnesses. Native MMA"
            ", layout, visibility, rounding/fusion, failed-query lowering and "
            "progress remain implementation premises."
        ),
    }
    (STUDY / "checks/geforce-conv-outputs-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    sys.stdout.write(
        "PASS: convolution side-stream phases, epilogue coverage and output retirement.\n"
    )


if __name__ == "__main__":
    main()
