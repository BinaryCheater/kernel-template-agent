"Replay mixed-input conversion contracts, prepack order and pure frontend guards."

from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import math
import struct
import sys
from pathlib import Path

from frontend_contract import DTYPES, Symbols

STUDY = Path(__file__).resolve().parents[1]
PREFIX = "examples/python/CuTeDSL/cute/blackwell/kernel/mixed_input_gemm/"


class MixedSymbols(Symbols):
    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        text = ast.unparse(node)
        if text in {
            "cutlass.cute.nvgpu.OperandMajorMode.K",
            "tcgen05.OperandSource.TMEM",
            "tcgen05.OperandSource.SMEM",
        }:
            return ast.copy_location(ast.Constant(node.attr), node)
        if isinstance(node.value, ast.Name) and node.value.id == "mixed_input_utils":
            return ast.copy_location(ast.Name(node.attr, ast.Load()), node)
        return super().visit_Attribute(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if ast.unparse(node.func) == "cutlass.target_version":
            assert len(node.keywords) == 1 and node.keywords[0].value.value == "13.1"
            return ast.copy_location(ast.Name("compiler_gate", ast.Load()), node)
        if ast.unparse(node.func) == "cutlass.const_expr":
            return self.visit(node.args[0])
        return self.generic_visit(node)


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    paths = [PREFIX + name for name in ("mixed_input_gemm.py", "mixed_input_host_utils.py")]
    paths += [
        "python/CuTeDSL/cutlass/utils/mixed_input_helpers.py",
        "python/CuTeDSL/cutlass/cute/arch/numeric_conversion.py",
    ]
    paths.append("python/CuTeDSL/cutlass/base_dsl/typing.py")
    for path in paths:
        entry = next(x for x in manifest["files"] if x["path"] == path)
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == entry["sha256"]
    helper = (root / paths[2]).read_text()
    kernel = (root / paths[0]).read_text()
    host = (root / paths[1]).read_text()
    DTYPES.update(Int4=4, Int8=8, Uint8=8)
    selected_funcs = {
        "is_valid_scale_granularity",
        "is_valid_tensor_alignment",
        "is_valid_mma_tiler_and_cluster_shape",
        "is_shuffle_a",
        "get_transform_a_source",
        "get_divisibility",
    }
    funcs = [
        copy.deepcopy(x)
        for x in ast.parse(helper).body
        if isinstance(x, ast.FunctionDef) and x.name in selected_funcs
    ]
    cls = copy.deepcopy(next(x for x in ast.parse(kernel).body if isinstance(x, ast.ClassDef)))
    cls.body = [
        x
        for x in cls.body
        if isinstance(x, ast.FunctionDef)
        and x.name in {"can_implement", "is_valid_epilog_store_option"}
    ]
    ns: dict = {"widths": DTYPES, "compiler_gate": False, "log2": math.log2}
    module = ast.Module(body=[*funcs, cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(MixedSymbols().visit(module)), paths[0], "exec"), ns)
    kind = ns[cls.name]

    def accepts(shape=(128, 128, 256, 1), dtype="Int4", gm=1, gk=256, store=True, cluster=(1, 1)):
        return kind.can_implement(
            shape,
            dtype,
            "BFloat16",
            "BFloat16",
            "k",
            "k",
            "n",
            gm,
            gk,
            (128, 128, 128),
            cluster,
            False,
            store,
        )

    assert accepts()
    assert accepts(dtype="Int8", gm=0, gk=0)
    assert accepts(dtype="Uint8", gm=0, gk=0)
    assert not accepts(gm=0, gk=0)
    assert not accepts(gm=2)
    assert not accepts(shape=(128, 128, 384, 1), gk=192)
    assert not accepts(dtype="Int8")
    assert accepts(shape=(136, 144, 256, 1), store=True)
    assert not accepts(shape=(136, 144, 256, 1), store=False)
    assert accepts(store=False)
    assert accepts(shape=(128, 128, 0, 1))
    try:
        ns["get_divisibility"](0)
    except ValueError:
        pass
    else:
        raise AssertionError("Source alignment helper must expose log2(0) for empty K")
    assert accepts(gk=-128)  # guard does not enforce positive scale extent
    assert accepts(dtype="Int8", gm=0, gk=0, cluster=(1, 32))
    assert ns["get_transform_a_source"]("K") == "TMEM"
    assert ns["get_transform_a_source"]("MN") == "SMEM"
    assert not ns["is_shuffle_a"]("k", 256, "Int4", "BFloat16", 256)
    ns["compiler_gate"] = True
    assert ns["is_shuffle_a"]("k", 256, "Int4", "BFloat16", 256)
    assert not ns["is_shuffle_a"]("m", 256, "Int4", "BFloat16", 256)
    assert not ns["is_shuffle_a"]("k", 256, "Int4", "Float16", 256)

    types_tree = ast.parse((root / "python/CuTeDSL/cutlass/base_dsl/typing.py").read_text())
    for name, bits, signed in (("Int4", 4, True), ("Int8", 8, True), ("Uint8", 8, False)):
        declaration = next(
            x for x in types_tree.body if isinstance(x, ast.ClassDef) and x.name == name
        )
        keywords = {
            x.arg: ast.literal_eval(x.value)
            for x in declaration.keywords
            if x.arg in {"width", "signed"}
        }
        assert keywords == {"width": bits, "signed": signed}
    conversions = 0
    for bits, signed in ((4, True), (8, True), (8, False)):
        for raw in range(2**bits):
            value = raw - 2**bits if signed and raw >= 2 ** (bits - 1) else raw
            assert int.from_bytes(raw.to_bytes(1), "little") == raw
            f16 = struct.unpack("<e", struct.pack("<e", value))[0]
            f32_bits = int.from_bytes(struct.pack("<f", value), "little")
            bf16 = struct.unpack("<f", (f32_bits & 0xFFFF0000).to_bytes(4, "little"))[0]
            assert f16 == bf16 == value
            conversions += 2
    assert conversions == 1056
    assert 0 != 0 - 8  # Offset-binary old decoder cannot supply signed Int4 raw0.
    perm_assign = next(
        x
        for x in ast.walk(ast.parse(host))
        if isinstance(x, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "perm" for t in x.targets)
    )
    perm = ast.literal_eval(perm_assign.value.args[0])
    assert perm == [0, 2, 1, 3, 4, 6, 5, 7]
    assert [perm[i] for i in perm] == list(range(8))
    packets = 0
    for position, raw in itertools.product(range(8), range(16)):
        logical = list(range(8))
        logical[position] = raw
        stored = [logical[i] for i in perm]
        restored = [stored[i] for i in perm]
        expected = [x if x < 8 else x - 16 for x in logical]
        actual = [x if x < 8 else x - 16 for x in restored]
        assert actual == expected
        packets += 1
    assert [list(range(8))[i] for i in perm] != list(range(8))
    intrinsic = (root / paths[3]).read_text()
    assert "the input elements are shuffled to (0, 2, 1, 3, 4, 6, 5, 7)" in intrinsic
    assert "tensor_transformed = tensor_transformed * scale" in kernel
    assert "tensor_transformed = mixed_input_utils.cvt_tensor_a(" in kernel
    assert "a_scales = a_scales.random_(-3, 3)" in host
    result = {
        "outcome": "PASS",
        "integer_code_type_pairs": conversions,
        "distinctive_shuffle_packets": packets,
        "host_permutation": perm,
        "compiler_gate": "Symbolic>=13.1 predicate, not an installed compiler probe",
        "negative_scale_extent_guard_gap": True,
        "empty_k_frontend_passes_but_host_divisibility_fails": True,
        "cluster_guard_not_a_capability_check": True,
        "host_scales_randomized_after_range_computation": True,
        "offset_binary_counterexample": {"raw": 0, "signed_value": 0, "old_q_minus_8": -8},
        "boundary": (
            "Exact integer/order and retained pure predicates only; no native "
            "converter, lane map, pipeline or numeric scale reassociation exec"
            "uted."
        ),
    }
    (STUDY / "checks/mixed-input-widening-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    sys.stdout.write(
        "PASS: mixed-input exact widening, paired shuffle and batched frontend guards.\n"
    )


if __name__ == "__main__":
    main()
