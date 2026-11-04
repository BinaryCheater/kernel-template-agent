"Replay grouped mixed-input scalar search, output masks and ordered pair credits."

from __future__ import annotations

import ast
import bisect
import copy
import hashlib
import itertools
import json
import struct
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import yaml
from frontend_contract import DTYPES
from mixed_input_widening import MixedSymbols

from kta.core.expr import Expr
from kta.core.schemas.knowledge import Template

STUDY = Path(__file__).resolve().parents[1]
PREFIX = "examples/python/CuTeDSL/cute/blackwell/kernel/mixed_input_gemm/"


class GroupSymbols(MixedSymbols):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.decorator_list = []
        return super().visit_FunctionDef(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if ast.unparse(node.func) == "cutlass.Int32":
            return self.visit(node.args[0])
        if ast.unparse(node.func) == "cute.ceil_div":
            a, b = map(self.visit, node.args)
            return ast.copy_location(
                ast.BinOp(
                    ast.BinOp(a, ast.Add(), ast.BinOp(b, ast.Sub(), ast.Constant(1))),
                    ast.FloorDiv(),
                    copy.deepcopy(b),
                ),
                node,
            )
        return super().visit_Call(node)


def bf16(value: Fraction) -> Fraction:
    bits = int.from_bytes(struct.pack("<f", float(value)), "little")
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return Fraction(struct.unpack("<f", rounded.to_bytes(4, "little"))[0])


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    names = ["grouped_mixed_input_gemm.py", "grouped_mixed_input_gemm_acc_scale.py"]
    helper_path = "python/CuTeDSL/cutlass/utils/mixed_input_helpers.py"
    for path in [
        *(PREFIX + name for name in names),
        PREFIX + "mixed_input_host_utils.py",
        helper_path,
    ]:
        entry = next(x for x in manifest["files"] if x["path"] == path)
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == entry["sha256"]
    helper_tree = ast.parse((root / helper_path).read_text())
    wanted = {
        "create_initial_contiguous_group_search_state",
        "contiguous_group_search",
        "is_valid_mma_tiler_and_cluster_shape",
        "is_valid_scale_granularity",
        "is_valid_tensor_alignment",
    }
    nodes = [
        copy.deepcopy(x)
        for x in helper_tree.body
        if isinstance(x, ast.FunctionDef) and x.name in wanted
    ]
    state = copy.deepcopy(
        next(
            x
            for x in helper_tree.body
            if isinstance(x, ast.ClassDef) and x.name == "ContiguousGGSearchState"
        )
    )
    state.body = [x for x in state.body if isinstance(x, ast.FunctionDef) and x.name == "__init__"]
    classes = []
    sources = []
    for name in names:
        source = (root / PREFIX / name).read_text()
        sources.append(source)
        tree = ast.parse(source)
        full = next(
            node.test
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and ast.unparse(node.test)
            == "work_tile.distance_to_boundary >= self.cta_tile_shape_mnk[1]"
        )
        tail = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Compare)
            and ast.unparse(node) == "m_thr_slice[i][1] < work_tile.distance_to_boundary"
        )
        full_expr = (
            ast.unparse(full)
            .replace("work_tile.distance_to_boundary", "remaining")
            .replace("self.cta_tile_shape_mnk[1]", "width")
        )
        tail_expr = (
            ast.unparse(tail)
            .replace("m_thr_slice[i][1]", "local")
            .replace("work_tile.distance_to_boundary", "remaining")
        )
        for remaining, local in itertools.product(range(-16, 33), range(16)):
            env = {"remaining": remaining, "local": local, "width": 16}
            assert eval(full_expr, {"__builtins__": {}}, env) == (remaining >= 16)
            assert eval(tail_expr, {"__builtins__": {}}, env) == (local < remaining)
        cls = copy.deepcopy(next(x for x in ast.parse(source).body if isinstance(x, ast.ClassDef)))
        cls.body = [
            x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "can_implement"
        ]
        classes.append(cls)
    DTYPES.update(Int4=4, Int8=8, Uint8=8)
    ns: dict = {"widths": DTYPES}
    module = ast.Module(body=[state, *nodes, *classes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(GroupSymbols().visit(module)), helper_path, "exec"), ns)
    for cls in classes:
        args = [
            (256, 16, 512, 4),
            "Int4",
            "BFloat16",
            "BFloat16",
            "k",
            "k",
            "n",
            1,
            256,
            (256, 16, 128),
            (2, 1),
            True,
        ]
        assert ns[cls.name].can_implement(*args)
        assert not ns[cls.name].can_implement(*[*args[:8], 192, *args[9:]])

    cursor_domains = cursor_queries = 0
    for groups, pattern, width, peers, stride in itertools.product(
        (1, 2, 5, 33), range(5), (8, 16, 32), (1, 2, 4), (1, 2, 7)
    ):
        lengths = [8 * ((g * (pattern + 1) + pattern) % 10) for g in range(groups)]
        prefix = [0, *itertools.accumulate(lengths)]
        unit = width * peers
        counts = [(length + unit - 1) // unit for length in lengths]
        work_prefix = list(itertools.accumulate(counts))
        for worker in range(stride):
            current = ns["create_initial_contiguous_group_search_state"]()
            for query in range(worker, sum(counts), stride):
                current = ns["contiguous_group_search"](
                    (128, unit, 128), groups, query, current, prefix, 1
                )
                g = bisect.bisect_right(work_prefix, query)
                local = query - (work_prefix[g - 1] if g else 0)
                assert current.cur_group_idx == g + 1
                assert current.cur_start == prefix[g] + local * unit
                assert current.cur_boundary == prefix[g + 1]
                # M-inner scheduling can ask for the same N work coordinate again.
                repeated = ns["contiguous_group_search"](
                    (128, unit, 128), groups, query, current, prefix, 1
                )
                assert vars(repeated) == vars(current)
                current = repeated
                cursor_queries += 2
            terminal = ns["contiguous_group_search"](
                (128, unit, 128), groups, sum(counts), current, prefix, 1
            )
            assert terminal.cur_group_idx > groups
            cursor_domains += 1

    lengths, prefix = [8, 0, 40, 16], [0, 8, 8, 48, 64]
    coverage_domains = inactive_peers = tma_items = simt_items = 0
    for rows, peers in itertools.product((128, 136, 256), (1, 2, 4)):
        width, unit = 16, 16 * peers
        counts = [(length + unit - 1) // unit for length in lengths]
        work = []
        for g, count in enumerate(counts):
            for local, peer in itertools.product(range(count), range(peers)):
                origin = prefix[g] + local * unit + peer * width
                remaining = prefix[g + 1] - origin
                work.append((g, origin, remaining))
        actual = Counter()
        wrong = Counter()
        for row_tile in range((rows + 127) // 128):
            for g, origin, remaining in work:
                inactive_peers += int(remaining <= 0)
                tma_items += int(remaining >= width)
                simt_items += int(0 < remaining < width)
                for r, n in itertools.product(range(128), range(width)):
                    row, column = row_tile * 128 + r, origin + n
                    if row < rows and n < remaining:
                        actual[(row, column)] += 1
                        assert prefix[g] <= column < prefix[g + 1]
                    if row < rows and column < prefix[-1]:
                        wrong[(row, column)] += 1
        assert len(actual) == rows * prefix[-1] and set(actual.values()) == {1}
        assert any(count > 1 for count in wrong.values())
        coverage_domains += 1
    assert sum((length + 15) // 16 for length in lengths) == 5
    assert (sum(lengths) + 15) // 16 == 4
    assert [0 < remaining for remaining in (8, -8)] == [True, False]

    credit_domains = 0
    for items, stages, batch in itertools.product(range(66), range(2, 9), (1, 2)):
        active = {}
        published = []
        waits = 0
        for first in range(0, items, batch):
            last = min(first + batch, items) - 1
            retired_prefix = last - stages
            active = {slot: item for slot, item in active.items() if item > retired_prefix}
            waits += 1
            for item in range(first, last + 1):
                slot = item % stages
                assert slot not in active
                active[slot] = item
                published.append(item)
        assert published == list(range(items))
        assert waits == (items + batch - 1) // batch
        credit_domains += 1
    # Last old reader1 returned, but reader0 remains: waiting for1 cannot protect slot0.
    retired_out_of_order, earlier_old, later_old = {1}, 0, 1
    assert later_old in retired_out_of_order and earlier_old not in retired_out_of_order
    assert "producer_acquire(trans2mma_producer_state)" in sources[1]
    assert "producer_acquire(trans2mma_producer_state0)" in sources[1]
    assert "if is_tile_cnt_odd:" in sources[1]
    assert "pipeline_state_clone = pipeline_state.clone()" in sources[1]

    # Same exact-real scale partition, different BF16 multiplication/FP32-promotion policy.
    scale0 = Fraction(145, 128)
    assert bf16(scale0) == scale0
    pre = bf16(7 * scale0) * 1 + bf16(-7 * 1) * scale0
    post = 7 * scale0 + (-7 * scale0) * 1
    assert pre == Fraction(1, 128) and post == 0
    assert bf16(pre) != bf16(post)
    numeric_cases = 0
    for row, n in itertools.product(range(3), range(3)):
        raw = [(k + row) % 16 - 8 for k in range(512)]
        b = [(2 * k + n) % 9 - 4 for k in range(512)]
        scales = [Fraction(row + 1, 2), Fraction(row + 3, 4)]
        direct = sum(raw[k] * b[k] * scales[k // 256] for k in range(512))
        promoted = sum(
            sum(raw[k] * b[k] for k in range(g * 256, (g + 1) * 256)) * scales[g] for g in range(2)
        )
        assert direct == promoted
        numeric_cases += 1
    assert "tTR_rAcc_final.fill(0.0)" in sources[1]
    assert "num_k_tiles_per_scale" in sources[1]
    reuse = Template.model_validate(
        yaml.safe_load(
            (
                STUDY.parent / "deepgemm/delta/block_scaled_partial_accumulation@1/template.yaml"
            ).read_text()
        )
    )
    variables = {"partial_k": 256, "gran_k_a": 256, "gran_k_b": 512, "gran_m": 1, "gran_n": 64}
    p = {
        "gran_k_a": 256,
        "gran_k_b": 512,
        "gran_m": 1,
        "gran_n": 64,
        "k_origin": 0,
        "K": 512,
        "neutral_tail_padding": False,
        "block_scaled_inputs": True,
        "numeric_contract_allows_partial_scaling": True,
    }
    for expression in [*reuse.constraints, *reuse.preconditions]:
        assert Expr.parse(expression.expr, variables.keys()).eval(variables, p=p)
    assert not Expr.parse(reuse.scope.predicate, variables.keys()).eval(
        variables, p={**p, "numeric_contract_allows_partial_scaling": False}
    )
    waits = Template.model_validate(
        yaml.safe_load((STUDY / "delta/ordered_slot_wait_coalescing@1/template.yaml").read_text())
    )
    assert (
        Expr.parse(waits.effects.transactions[0].count, ["batch_width"]).eval(
            {"batch_width": 2}, p={"items": 3}
        )
        == 2
    )
    result = {
        "outcome": "PASS",
        "source_cursor_domains": cursor_domains,
        "source_cursor_queries": cursor_queries,
        "segment_output_coverage_domains": coverage_domains,
        "inactive_peer_instances": inactive_peers,
        "full_tma_items": tma_items,
        "partial_simt_items": simt_items,
        "separate_vs_global_tile_counts": [5, 4],
        "pair_credit_domains": credit_domains,
        "out_of_order_credit_counterexample": {"retired_old": [1], "live_old": 0},
        "exact_scale_partition_cases": numeric_cases,
        "bf16_scale_reassociation_counterexample": {
            "scale": str(scale0),
            "input_scaled_result": str(pre),
            "partial_promoted_result": str(post),
            "difference_survives_bf16_output": True,
        },
        "boundary": (
            "Retained scalar search/predicates and rational/order models only;"
            " native group clipping, prefix retirement, lane maps and numeric "
            "tolerance remain open."
        ),
    }
    binding = {
        "group_lengths": lengths,
        "prefix": prefix,
        "mnkl_host": [256, 16, 512, 4],
        "mma_mnk": [256, 16, 128],
        "cluster_mn": [2, 1],
        "dtype_a_b_c_acc": ["Int4", "BFloat16", "BFloat16", "Float32"],
        "scale_granularity_m_k": [1, 256],
        "decoder": "segmented_axis_work_decode@1",
        "store_variant": {
            "name": "group_boundary_hybrid",
            "tma_domain": "remaining>=CTA_N",
            "simt_domain": "0<remaining<CTA_N with M/N predicates",
            "inactive": "remaining<=0 executes collective work but stores no elements",
            "stage_clock": "advance output ring only for TMA-eligible subtiles",
        },
        "partial_scale_reuse": {
            "identity": "block_scaled_partial_accumulation@1",
            "variables": variables,
            "p": p,
            "constant_b_scale": (
                "Logical unit scale over the full K512/fusedN64 domain; no physica"
                "l B-scale tensor is allocated."
            ),
        },
    }
    (STUDY / "inputs/grouped-mixed-binding.json").write_text(json.dumps(binding, indent=2) + "\n")
    (STUDY / "checks/grouped-mixed-input-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    sys.stdout.write(
        "PASS: segmented source cursor, hybrid output coverage, pair credi"
        "ts and conditional scale promotion.\n"
    )


if __name__ == "__main__":
    main()
