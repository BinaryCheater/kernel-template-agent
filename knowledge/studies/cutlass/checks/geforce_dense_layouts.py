"Replay selected static scheduler, epilogue partition metadata and host fixtures."

from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

from dense_contracts import DenseSymbols
from frontend_contract import DTYPES
from moe_dense import ScalarSource

STUDY = Path(__file__).resolve().parents[1]
ENTRY = "examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py"


def tuple_size(value, mode=None):
    if mode is not None:
        for axis in mode:
            value = value[axis]
    return math.prod(tuple_size(x) for x in value) if isinstance(value, tuple) else value


def compile_functions(functions, namespace):
    module = ast.Module(copy.deepcopy(functions), [])
    module = ScalarSource().visit(module)
    for node in module.body:
        node.decorator_list = []
    exec(compile(ast.fix_missing_locations(module), "retained-scalars", "exec"), namespace)


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    files = [
        ENTRY,
        "python/CuTeDSL/cutlass/utils/static_persistent_tile_scheduler.py",
        "python/CuTeDSL/cutlass/utils/hopper_helpers.py",
        "python/CuTeDSL/cutlass/torch.py",
        "python/CuTeDSL/cutlass/cute/nvgpu/warp/copy.py",
    ]
    trees = {}
    for path in files:
        data = (Path(manifest["tree"]) / path).read_bytes()
        record = next(x for x in manifest["files"] if x["path"] == path)
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
        trees[Path(path).name] = ast.parse(data)
    scheduler = trees["static_persistent_tile_scheduler.py"]
    classes = {x.name: x for x in scheduler.body if isinstance(x, ast.ClassDef)}
    params = classes["PersistentTileSchedulerParams"]
    sched = classes["StaticPersistentTileScheduler"]
    methods = {x.name: x for x in sched.body if isinstance(x, ast.FunctionDef)}
    grid_fn = next(
        x for x in params.body if isinstance(x, ast.FunctionDef) and x.name == "get_grid_shape"
    )
    ns = {"tuple_size": tuple_size}
    compile_functions(
        [
            grid_fn,
            *(
                methods[x]
                for x in (
                    "create",
                    "__init__",
                    "_get_cluster_work_idx_with_fastdivmod",
                    "advance_to_next_work",
                )
            ),
        ],
        ns,
    )
    kind = type("RetainedScalarScheduler", (), {"__init__": ns["__init__"]})
    ns["StaticPersistentTileScheduler"] = kind
    scheduler_domains = scheduled_coordinates = 0
    for tm, tn, batches, active in itertools.product(
        (1, 2, 5, 17), (1, 3, 7), (1, 2, 3), (1, 2, 8, 64)
    ):
        total = tm * tn * batches
        p = SimpleNamespace(
            problem_layout_ncluster_mnl=SimpleNamespace(shape=(tm, tn, batches)),
            cluster_shape_mn=(1, 1),
            _has_distinct_fallback=False,
            cluster_shape_major_fdd=tm,
            cluster_shape_minor_fdd=tn,
            raster_along_m=True,
        )
        grid = ns["get_grid_shape"](p, active)
        assert grid == (1, 1, min(total, active))
        owned = set()
        for z in range(grid[2]):
            state = ns["create"](p, (0, 0, z), grid)
            while state._current_work_linear_idx < total:
                q = state._current_work_linear_idx
                coord = ns["_get_cluster_work_idx_with_fastdivmod"](state, q)
                assert coord == (q % tm, (q // tm) % tn, q // (tm * tn))
                assert coord not in owned
                owned.add(coord)
                ns["advance_to_next_work"](state)
            assert state._num_tiles_executed == len(range(z, total, grid[2]))
        assert owned == set(itertools.product(range(tm), range(tn), range(batches)))
        scheduled_coordinates += len(owned)
        scheduler_domains += 1

    helpers = trees["hopper_helpers.py"]
    epi = next(
        x
        for x in helpers.body
        if isinstance(x, ast.FunctionDef) and x.name == "compute_tile_shape_or_override"
    )
    transformed = DenseSymbols().visit(ScalarSource().visit(ast.Module([copy.deepcopy(epi)], [])))
    ns = {"tuple_size": tuple_size, "widths": DTYPES}
    exec(compile(ast.fix_missing_locations(transformed), "source-epi", "exec"), ns)
    menu = json.loads((STUDY / "checks/geforce-dense-result.json").read_text())["tile_menu"]
    epi_domains = epi_coordinates = 0
    for tile, output in itertools.product(menu, ("Float16", "BFloat16", "Float32", "Float8E4M3FN")):
        em, en = ns["compute_tile_shape_or_override"](tuple(tile), output)
        assert (em, en) == (64, 64 if DTYPES[output] == 8 else 32)
        assert tile[0] % em == 0 and tile[1] % en == 0
        coords = [
            (mi * em + m, ni * en + n)
            for mi, ni, m, n in itertools.product(
                range(tile[0] // em), range(tile[1] // en), range(em), range(en)
            )
        ]
        assert len(set(coords)) == len(coords) == tile[0] * tile[1]
        epi_coordinates += len(coords)
        epi_domains += 1

    # Output store operation selected by width; the geometric StMatrix atom in
    # dense_gemm.py remains separate and requires native compatibility checking.
    store = next(
        x for x in helpers.body if isinstance(x, ast.FunctionDef) and x.name == "get_smem_store_op"
    )
    copied = copy.deepcopy(store)
    copied.body = [
        x
        for x in copied.body
        if not isinstance(x, ast.FunctionDef)
        and not (
            isinstance(x, ast.Expr)
            and isinstance(x.value, ast.Call)
            and ast.unparse(x.value.func) == "validate_type"
        )
    ]
    ns = {
        "cute": SimpleNamespace(make_copy_atom=lambda op, dtype, **kw: op),
        "StMatrix8x8x16bOp": lambda *args: "StMatrix16",
        "CopyUniversalOp": lambda: "Universal",
    }
    compile_functions([copied], ns)
    stores = {}
    for output, major in itertools.product(
        ("Float16", "BFloat16", "Float32", "Float8E4M3FN"), (False, True)
    ):
        selected = ns["get_smem_store_op"](
            SimpleNamespace(is_m_major_c=lambda major=major: major),
            SimpleNamespace(width=DTYPES[output]),
            SimpleNamespace(width=32),
        )
        assert selected == ("StMatrix16" if DTYPES[output] == 16 else "Universal")
        stores[f"{output}_m_major_{major}"] = selected

    # Pin the actual fixture generator's initialization defaults, not its caller's
    # inaccurate random-range comment. No random samples are needed to recover it.
    matrix = next(
        x for x in trees["torch.py"].body if isinstance(x, ast.FunctionDef) and x.name == "matrix"
    )
    defaults = next(
        x
        for x in matrix.body
        if isinstance(x, ast.If) and "init_config is None" in ast.unparse(x.test)
    )
    env = {
        "torch_dtype": SimpleNamespace(is_signed=True),
        "RandomInitConfig": lambda **kwargs: kwargs,
    }
    exec(
        compile(ast.Module(copy.deepcopy(defaults.body), []), "source-fixture-default", "exec"), env
    )
    assert env["init_config"] == {"min_val": -2, "max_val": 2}
    result = {
        "outcome": "PASS",
        "scheduler_domains": scheduler_domains,
        "scheduled_coordinates": scheduled_coordinates,
        "epilogue_domains": epi_domains,
        "epilogue_coordinates": epi_coordinates,
        "selected_output_stores": stores,
        "fixture_integer_values": [-2, -1, 0, 1],
        "boundary": (
            "Retained static create/grid/advance/decode and helper metadata on"
            " scalar divisors; symbolic store selection. Native FastDivmod, St"
            "Matrix typed partition, TMA bounds, aligned allocation and numeri"
            "cal precision remain implementation obligations."
        ),
    }
    (STUDY / "checks/geforce-dense-layouts-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    sys.stdout.write(
        "PASS: GeForce dense static ownership, epilogue coordinates and store selection.\n"
    )


if __name__ == "__main__":
    main()
