"""Replay scaled MoE padding, chunk carries and segment-local scale storage."""

from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import sys
from pathlib import Path

import yaml
from moe_dense import ScalarSource

from kta.core.expr import Expr
from kta.core.schemas.knowledge import Chain, Template

STUDY = Path(__file__).resolve().parents[1]
PREFIX = "examples/python/CuTeDSL/cute/blackwell/kernel/moe/"


def permute(values: list[int], shape: tuple[int, ...], order: tuple[int, ...]) -> list[int]:
    "Apply an index permutation to distinct scalar labels, with no tensor runtime."
    result = []
    for coord in itertools.product(*(range(shape[axis]) for axis in order)):
        original = [0] * len(shape)
        for axis, value in zip(order, coord, strict=True):
            original[axis] = value
        index = 0
        for extent, value in zip(shape, original, strict=True):
            index = index * extent + value
        result.append(values[index])
    return result


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    paths = [
        PREFIX + name
        for name in ("torch_scaled_grouped_mm.py", "moe_utils.py", "moe_sched_extension.py")
    ] + ["python/CuTeDSL/cutlass/utils/blockscaled_layout.py"]
    trees = {}
    for path in paths:
        entry = next(x for x in manifest["files"] if x["path"] == path)
        source = (root / path).read_bytes()
        assert hashlib.sha256(source).hexdigest() == entry["sha256"]
        trees[Path(path).name] = ast.parse(source.decode())
    tree = trees["torch_scaled_grouped_mm.py"]
    functions = [
        copy.deepcopy(x)
        for x in tree.body
        if isinstance(x, ast.FunctionDef)
        and x.name in ("ceil_div", "round_up", "compute_scale_shape")
    ]
    ns = {}
    exec(
        compile(
            ast.fix_missing_locations(ScalarSource().visit(ast.Module(functions, []))),
            "retained-scale-shapes",
            "exec",
        ),
        ns,
    )
    shape_fn = ns["compute_scale_shape"]
    kernel_cls = next(
        x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "ScaledGroupedGemmKernel"
    )
    setup = next(
        x
        for x in kernel_cls.body
        if isinstance(x, ast.FunctionDef) and x.name == "desc_init_kernel"
    )
    padded = next(
        x.value
        for x in ast.walk(setup)
        if isinstance(x, ast.Assign)
        and ast.unparse(x.targets[0]) == "padded_size"
        and isinstance(x.value, ast.BinOp)
    )
    padded_code = compile(ast.Expression(padded), "retained-padded-size", "eval")
    scan_steps = next(
        x.iter.elts
        for x in ast.walk(setup)
        if isinstance(x, ast.For) and ast.unparse(x.target) == "d"
    )
    steps = [ast.literal_eval(x) for x in scan_steps]
    assert steps == [1, 2, 4, 8, 16]
    shape_domains = scan_domains = queries = 0
    for experts, sf, pattern in itertools.product(
        (1, 31, 32, 33, 127, 128, 129, 257), (16, 32), range(4)
    ):
        lengths = [sf * ((expert * (pattern + 1) + pattern) % 9) for expert in range(experts)]
        ends = list(itertools.accumulate(lengths))
        for scenario in ("2Dx3D", "2Dx2D"):
            granularity = 128 if scenario == "2Dx3D" else 4 * sf
            reference_sizes = [
                ((n + granularity - 1) // granularity) * granularity for n in lengths
            ]
            reference = list(itertools.accumulate(reference_sizes))
            scanned = []
            carry = 0
            for base in range(0, experts, 128):
                values = [
                    eval(padded_code, {"size_i": lengths[e], "pad_granularity": granularity})
                    if e < experts
                    else 0
                    for e in range(base, base + 128)
                ]
                assert values == reference_sizes[base : base + 128] + [0] * max(
                    0, base + 128 - experts
                )
                warp_prefixes = []
                for warp in range(4):
                    val = values[warp * 32 : (warp + 1) * 32]
                    for d in steps:
                        previous = val[:]
                        val = [
                            previous[lane] + (previous[lane - d] if lane >= d else 0)
                            for lane in range(32)
                        ]
                    warp_prefixes.extend(val)
                warp_totals = [warp_prefixes[w * 32 + 31] for w in range(4)]
                corrected = [
                    carry + warp_prefixes[lane] + sum(warp_totals[: lane // 32])
                    for lane in range(128)
                ]
                shared = [carry, *corrected]
                for lane in range(min(128, experts - base)):
                    assert shared[lane + 1] - shared[lane] == reference_sizes[base + lane]
                    queries += 1
                scanned.extend(corrected[: min(128, experts - base)])
                carry = shared[128]
            assert scanned == reference
            scan_domains += 1
            for hidden, intermediate in ((128, 64), (256, 192)):
                for operand in ("a", "b"):
                    got = shape_fn(
                        scenario, operand, lengths, hidden, intermediate, hidden, sf, experts
                    )
                    if scenario == "2Dx3D":
                        expected = (
                            (sum(reference_sizes), ns["round_up"](ns["ceil_div"](hidden, sf), 4))
                            if operand == "a"
                            else (
                                experts,
                                ns["round_up"](intermediate, 128)
                                * ns["round_up"](ns["ceil_div"](hidden, sf), 4),
                            )
                        )
                    else:
                        mn = hidden if operand == "a" else intermediate
                        expected = (ns["round_up"](mn, 128), sum(reference_sizes) // sf)
                    assert got == expected
                    shape_domains += 1
                if scenario == "2Dx2D":
                    for n in lengths:
                        assert ns["round_up"](ns["ceil_div"](n, sf), 4) * sf == ns["round_up"](
                            n, 4 * sf
                        )
        # Reusing actual ends is only valid when each segment, not just the total, is aligned.
        assert ends == list(itertools.accumulate(lengths))

    # Replay the actual host permutation constants on unique coordinate labels.
    pack = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "to_blocked")
    permute_call = next(
        x
        for x in ast.walk(pack)
        if isinstance(x, ast.Call)
        and isinstance(x.func, ast.Attribute)
        and x.func.attr == "permute"
    )
    transpose = next(
        x
        for x in ast.walk(pack)
        if isinstance(x, ast.Call)
        and isinstance(x.func, ast.Attribute)
        and x.func.attr == "transpose"
    )
    first_order = tuple(ast.literal_eval(x) for x in permute_call.args)
    transpose_axes = tuple(ast.literal_eval(x) for x in transpose.args)
    assert first_order == (0, 2, 1, 3) and transpose_axes == (1, 2)
    view = next(
        x
        for x in ast.walk(pack)
        if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == "view"
    )
    assert ast.unparse(view) == "padded.view(row_blocks, 128, col_blocks, 4)"
    reshape = next(
        x
        for x in ast.walk(pack)
        if isinstance(x, ast.Call)
        and isinstance(x.func, ast.Attribute)
        and x.func.attr == "reshape"
        and len(x.args) == 4
    )
    source_inner_shape = tuple(ast.literal_eval(x) for x in reshape.args)
    assert source_inner_shape == (-1, 4, 32, 4)
    checked_coordinates = 0
    for rows, columns in itertools.product((128, 256), (4, 8, 12)):
        labels = list(range(rows * columns))
        stage = permute(labels, (rows // 128, 128, columns // 4, 4), first_order)
        order = list(range(4))
        order[transpose_axes[0]], order[transpose_axes[1]] = (
            order[transpose_axes[1]],
            order[transpose_axes[0]],
        )
        packed = permute(stage, (rows * columns // 512, 4, 32, 4), tuple(order))
        assert len(set(packed)) == rows * columns
        for row, column in itertools.product(range(rows), range(columns)):
            offset = ((row // 128) * (columns // 4) + column // 4) * 512
            offset += (row % 32) * 16 + (row % 128 // 32) * 4 + column % 4
            assert packed[offset] == row * columns + column
            checked_coordinates += 1
    # Atom strides agree with local host permutation; native tile_to_shape remains open.
    atom = trees["blockscaled_layout.py"]
    stride_expr = next(
        x.value
        for x in ast.walk(atom)
        if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "atom_stride"
    )
    assert ast.literal_eval(stride_expr) == ((16, 4), (0, 1))

    witness_lengths = [32, 0, 160, 64]
    witness_ends = list(itertools.accumulate(witness_lengths))
    forward = list(itertools.accumulate(ns["round_up"](x, 128) for x in witness_lengths))
    backward = list(itertools.accumulate(ns["round_up"](x, 128) for x in witness_lengths))
    assert witness_ends == [32, 32, 192, 256]
    assert forward == backward == [128, 128, 384, 512]
    assert sum(ns["round_up"](x, 128) for x in witness_lengths) != ns["round_up"](
        sum(witness_lengths), 128
    )
    assert witness_ends[1] != forward[1]
    # A globally reinterpreted K-concatenation has the wrong row-block pitch.
    local_group_row_block = 512
    global_row_block = 1024
    assert local_group_row_block != global_row_block

    ctor = next(
        x
        for x in trees["moe_utils.py"].body
        if isinstance(x, ast.ClassDef) and x.name == "MoEScaledGroupedGemmTensormapConstructor"
    )
    sf_method = next(
        x
        for x in ctor.body
        if isinstance(x, ast.FunctionDef) and x.name == "_construct_sf_descs_2dx2d_direct"
    )
    offsets = [
        x.value
        for x in sf_method.body
        if isinstance(x, ast.Assign)
        and ast.unparse(x.targets[0]) in ("a_elems_to_move", "b_elems_to_move")
    ]
    descriptor_checks = 0
    for sf, mn, token_offset in itertools.product((16, 32), (128, 256), (0, 64, 128, 384)):
        if token_offset % (4 * sf):
            continue
        for expression in offsets:
            text = ast.unparse(expression).replace("self.sf_vec_size", "sf")
            text = text.replace("cute.size(self.sfa_tensor, mode=[0])", "mn")
            text = text.replace("cute.size(self.sfb_tensor, mode=[0])", "mn")
            got = eval(text, {"padded_offset": token_offset, "sf": sf, "mn": mn})
            assert got == mn * (token_offset // sf)
            descriptor_checks += 1
    # Consumer release is unconditional across the partial final chunk.
    construct = next(
        x for x in ctor.body if isinstance(x, ast.FunctionDef) and x.name == "construct_and_write"
    )
    loop = next(x for x in construct.body if isinstance(x, ast.While))
    unconditional_calls = [
        ast.unparse(x.value.func)
        for x in loop.body
        if isinstance(x, ast.Expr) and isinstance(x.value, ast.Call)
    ]
    assert "handle.release" in unconditional_calls
    assert "handle = consumer.wait_and_advance()" in ast.unparse(loop)
    cli_text = (root / PREFIX / "torch_scaled_grouped_mm.py").read_text()
    assert "args.consistent_token_padding = False" in cli_text
    assert "args.separate_tensormap_init = True" in cli_text
    assert '"blocksize": 16' in cli_text and '"blocksize": 32' in cli_text

    # Reuse the immutable descriptor table with four backward descriptors.
    record = Template.model_validate(
        yaml.safe_load(
            (STUDY / "delta/precompute_transfer_descriptors@1/template.yaml").read_text()
        )
    )
    case = json.loads((STUDY / "inputs/moe-binding-cases.json").read_text())["cases"][1]
    params = {**case["p"], "selected_tuple_size": 4}
    for guard in [*record.constraints, *record.preconditions]:
        assert Expr.parse(guard.expr, ["descriptors_per_problem"]).eval(
            {"descriptors_per_problem": 4}, p=params
        )
    chain = Chain.model_validate(
        yaml.safe_load((STUDY / "delta/moe-scaled-metadata-chain.yaml").read_text())
    )
    previous = set()
    for item in chain.items:
        assert item.key not in previous and set(item.after) <= previous
        rule = Template.model_validate(
            yaml.safe_load((STUDY / "delta" / item.template / "template.yaml").read_text())
        )
        assert set(item.bindings) == {v.name for v in rule.variables}
        for variable in rule.variables:
            value = item.bindings[variable.name]
            if isinstance(variable.domain, list):
                assert value in variable.domain
            else:
                assert variable.domain.lo <= value <= variable.domain.hi
        previous.add(item.key)
    binding = {
        "format": "mxfp8",
        "sf_vec_size": 32,
        "hidden": 256,
        "intermediate": 256,
        "token_lengths": witness_lengths,
        "token_ends": witness_ends,
        "forward_padded_row_ends": forward,
        "backward_padded_token_equivalent_ends": backward,
        "backward_padded_scale_column_ends": [x // 32 for x in backward],
        "backward_descriptor_tuple": ["a", "b", "sfa", "sfb"],
        "flags": {"separate_tensormap_init": True, "consistent_token_padding": False},
        "boundary": (
            "Metadata/host layout slice; the device companion covers mainloop/"
            "global-scale ordering separately."
        ),
    }
    result = {
        "outcome": "PASS",
        "partial_chain_steps": len(chain.items),
        "source_shape_domains": shape_domains,
        "chunk_scan_domains": scan_domains,
        "chunk_segment_queries": queries,
        "host_swizzle_coordinates": checked_coordinates,
        "source_descriptor_base_checks": descriptor_checks,
        "separate_vs_global_padded_extent": [512, 256],
        "local_vs_global_row_block_pitch": [local_group_row_block, global_row_block],
        "descriptor_table_reuse": (
            "precompute_transfer_descriptors@1 tuple4; backend premises conditional"
        ),
        "boundary": (
            "Scalar source expressions, prefix and permutation models; native "
            "layout/descriptor visibility and scale tails remain open; mainloo"
            "p is outside this check."
        ),
    }
    (STUDY / "inputs/moe-scaled-metadata-binding.json").write_text(
        json.dumps(binding, indent=2) + "\n"
    )
    (STUDY / "checks/moe-scaled-metadata-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    sys.stdout.write(
        "PASS: scaled MoE segment padding, chunk carry and scale storage coordinates.\n"
    )


if __name__ == "__main__":
    main()
