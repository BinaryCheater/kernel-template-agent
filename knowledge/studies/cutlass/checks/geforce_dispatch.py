"Replay SM120 dispatch symbolically and the selected FP4 register bit placement."

from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from frontend_contract import DTYPES, Symbols

STUDY = Path(__file__).resolve().parents[1]
PREFIX = "examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/"


class DispatchSymbols(Symbols):
    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.value.id == "cutlass":
            return ast.copy_location(ast.Constant(node.attr), node)
        return super().visit_Attribute(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        name = ast.unparse(node.func)
        if name.startswith("cute.nvgpu.warp.") or name == "cute.make_copy_atom":
            # Retain branch/arguments as a constructor descriptor; no backend substitute executes.
            fields = [ast.Constant(name.split(".")[-1]), *[self.visit(x) for x in node.args]]
            fields.extend(
                ast.Tuple([ast.Constant(x.arg), self.visit(x.value)], ast.Load())
                for x in node.keywords
            )
            return ast.copy_location(ast.Tuple(fields, ast.Load()), node)
        return self.generic_visit(node)


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    paths = [
        PREFIX + name
        for name in (
            "blockscaled_gemm_dispatch.py",
            "dense_blockscaled_gemm_persistent_cooperative.py",
            "dense_blockscaled_gemm_persistent_pingpong.py",
        )
    ]
    paths += [
        "python/CuTeDSL/cutlass/cute/nvgpu/warp/mma.py",
        "include/cute/atom/mma_traits_sm120.hpp",
    ]
    data = {}
    for path in paths:
        content = (root / path).read_bytes()
        pinned = next(x for x in manifest["files"] if x["path"] == path)
        assert hashlib.sha256(content).hexdigest() == pinned["sha256"]
        data[Path(path).name] = content.decode()
    tree = ast.parse(data["blockscaled_gemm_dispatch.py"])
    selected = [copy.deepcopy(x) for x in tree.body if isinstance(x, (ast.FunctionDef, ast.Assign))]
    widths = {**DTYPES, "Int8": 8, "Float6E2M3FN": 6, "Float6E3M2FN": 6}
    pairs = {("Float4E2M1FN", x) for x in ("Float8E4M3FN", "Float8E5M2")}
    pairs |= {(b, a) for a, b in pairs}
    pair_source = next(
        x.value
        for x in ast.parse(data["mma.py"]).body
        if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "MXF8F6F4_SUPPORTED_PAIRS"
    )
    actual_pairs = {tuple(ast.unparse(x) for x in pair.elts) for pair in pair_source.args[0].elts}
    assert actual_pairs == pairs
    ns = {"widths": widths, "MXF8F6F4_SUPPORTED_PAIRS": pairs}
    exec(
        compile(
            ast.fix_missing_locations(DispatchSymbols().visit(ast.Module(selected, []))),
            "retained-sm120-dispatch",
            "exec",
        ),
        ns,
    )
    menus = {}
    for name in (
        "dense_blockscaled_gemm_persistent_cooperative.py",
        "dense_blockscaled_gemm_persistent_pingpong.py",
    ):
        kernel_tree = ast.parse(data[name])
        menus[name] = {
            ast.unparse(x.targets[0]): ast.literal_eval(x.value)
            for x in ast.walk(kernel_tree)
            if isinstance(x, ast.Assign)
            and ast.unparse(x.targets[0]) in ("fp4_allowed_tiles", "fp8_allowed_tiles")
        }
        assert menus[name]["fp8_allowed_tiles"] == {(128, 128, 128)}
    valid = invalid = 0
    dispatches = set()
    types = (
        "Float4E2M1FN",
        "Float8E4M3FN",
        "Float8E5M2",
        "Float6E2M3FN",
        "Float6E3M2FN",
        "Float16",
    )
    for a, b, sf, vec, tile, menu in itertools.product(
        types,
        types,
        ("Float8E4M3FN", "Float8E8M0FNU"),
        (16, 32, 64),
        ((128, 128, 128), (128, 128, 256), (128, 256, 64)),
        menus.values(),
    ):
        fp8 = ("Float8E4M3FN", "Float8E5M2")
        expected = (
            (
                a == b == "Float4E2M1FN"
                and tile in menu["fp4_allowed_tiles"]
                and (
                    (vec == 16 and sf == "Float8E4M3FN")
                    or (vec == 32 and sf == "Float8E8M0FNU" and tile[2] == 128)
                )
            )
            or (
                a == b
                and a in fp8
                and vec == 32
                and sf == "Float8E8M0FNU"
                and tile in menu["fp8_allowed_tiles"]
            )
            or ((a, b) in pairs and vec == 32 and sf == "Float8E8M0FNU" and tile == (128, 128, 128))
        )
        args = SimpleNamespace(
            a_dtype=a, b_dtype=b, sf_dtype=sf, sf_vec_size=vec, tile_shape_mnk=tile
        )
        try:
            ns["validate_blockscaled_args"](
                args, menu["fp4_allowed_tiles"], menu["fp8_allowed_tiles"]
            )
        except ValueError:
            assert not expected
            invalid += 1
        else:
            assert expected
            descriptor, perm = ns["make_sm120_blockscaled_mma_op"](a, b, "Float32", sf, vec)
            dispatches.add(descriptor[0])
            assert perm == (a != "Float4E2M1FN" or b != "Float4E2M1FN")
            valid += 1
    assert len(dispatches) == 4
    assert ns["is_mxfp8_k64_sf_k128_tile"](
        (128, 256, 64), "Float8E4M3FN", "Float8E4M3FN", "Float8E8M0FNU", 32
    )
    assert not ns["is_mxfp8_k64_sf_k128_tile"](
        (128, 256, 64), "Float4E2M1FN", "Float8E4M3FN", "Float8E8M0FNU", 32
    )
    for a, b in itertools.product(types, repeat=2):
        try:
            result = ns["make_sm120_blockscaled_mma_op"](a, b, "Float32", "Float8E8M0FNU", 32)
        except ValueError:
            assert not ((a == b and a in (*fp8, "Float4E2M1FN")) or (a, b) in pairs)
        else:
            assert (a == b and a in (*fp8, "Float4E2M1FN")) or (a, b) in pairs
    mixed_load = ns["make_ldmatrix_atom"]("Float4E2M1FN", False, mixed_mode=True)
    assert mixed_load[1][0] == "LdMatrix8x16x8bOp" and mixed_load[2] == "Int8"
    assert ("unpack_bits", 4) in mixed_load[1]
    try:
        ns["make_ldmatrix_atom"]("Float4E2M1FN", True, mixed_mode=True)
    except AssertionError:
        transpose_rejected = True
    else:
        raise AssertionError("Mixed FP4 unpack cannot transpose")
    code_cases = packet_cases = shift_sites = 0
    for shift, q in itertools.product(range(5), range(16)):
        byte = q << shift
        assert 0 <= byte <= 255 and (byte >> shift) & 15 == q
        code_cases += 1
    for a, b in itertools.product(range(16), repeat=2):
        payloads = [a, b, (a + 7) % 16, (b + 11) % 16]
        word = sum(q << (8 * position) for position, q in enumerate(payloads))
        encoded = word << ns["FP4_SHIFT_BITS"]
        recovered = [(encoded >> (8 * position + 2)) & 15 for position in range(4)]
        assert recovered == payloads
        packet_cases += 1
    assert ((10 << 2) >> 2) & 15 == 10
    assert (10 >> 2) & 15 != 10
    assert (((10 << 2) << 2) >> 2) & 15 != 10
    for name in menus:
        t = ast.parse(data[name])
        shifts = [
            x
            for x in ast.walk(t)
            if isinstance(x, ast.BinOp)
            and isinstance(x.op, ast.LShift)
            and ast.unparse(x.right) == "FP4_SHIFT_BITS"
        ]
        assert len(shifts) == 4
        for expression in shifts:
            view = expression.left.value.id
            value = eval(
                compile(ast.Expression(expression), "retained-shift", "eval"),
                {view: [10], "_i": 0, "FP4_SHIFT_BITS": 2},
            )
            assert value == 40
            shift_sites += 1
    cpp = data["mma_traits_sm120.hpp"]
    assert "0b0000ABCD --> 0b00ABCD00" in cpp and "return v << 2;" in cpp
    result = {
        "outcome": "PASS",
        "cli_valid_cases": valid,
        "cli_rejected_cases": invalid,
        "symbolic_mma_constructors": sorted(dispatches),
        "placement_code_cases": code_cases,
        "four_byte_packets": packet_cases,
        "selected_shift_sites": shift_sites,
        "mixed_transpose_rejected": transpose_rejected,
        "k64_helper_is_cli_selected": False,
        "boundary": (
            "Symbolic retained dispatch and bitfield source replay only; nativ"
            "e constructors, LDSM lane maps and persistent kernels remain open"
            "."
        ),
    }
    (STUDY / "checks/geforce-dispatch-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: SM120 dispatch menus, mixed unpack and code-preserving operand placement.\n"
    )


if __name__ == "__main__":
    main()
