"Evaluate retained dense-GEMM type/geometry predicates and stage/numeric witnesses."

from __future__ import annotations

import ast
import copy
import itertools
import json
import math
import struct
import sys
from pathlib import Path

from frontend_contract import DTYPES, Symbols

STUDY = Path(__file__).resolve().parents[1]
PREFIX = "examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/"


class DenseSymbols(Symbols):
    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.value.id == "testing":
            assert node.attr == "CantImplementError"
            return ast.copy_location(ast.Name("ValueError", ast.Load()), node)
        return super().visit_Attribute(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in ("Int8", "Uint8"):
            return ast.copy_location(ast.Constant(node.id), node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "check_int8_mma_arch":
            node.func = ast.Name("check_int8_mma_arch", ast.Load())
            node.keywords.append(ast.keyword("target_arch", ast.Constant("sm_100")))
        return self.generic_visit(node)


def main() -> None:
    DTYPES.update({"TFloat32": 32, "Int8": 8, "Uint8": 8, "Int32": 32})
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    helper_body = ast.parse(
        (root / "python/CuTeDSL/cutlass/utils/blackwell_helpers.py").read_text()
    )
    arch_guard = next(
        x
        for x in helper_body.body
        if isinstance(x, ast.FunctionDef) and x.name == "check_int8_mma_arch"
    )
    acc_inputs = {
        "Float32": {"Float16", "BFloat16", "TFloat32", "Float8E4M3FN", "Float8E5M2"},
        "Float16": {"Float16", "Float8E4M3FN", "Float8E5M2"},
        "Int32": {"Int8", "Uint8"},
    }
    acc_outputs = {
        "Float32": {
            "Float32",
            "Float16",
            "BFloat16",
            "Float8E4M3FN",
            "Float8E5M2",
            "Int32",
            "Int8",
            "Uint8",
        },
        "Float16": {"BFloat16", "Float16"},
        "Int32": {"BFloat16", "Float16", "Float32", "Int32", "Int8", "Uint8"},
    }
    counts = {}
    for file in sorted((root / PREFIX).glob("*.py")):
        tree = ast.parse(file.read_text())
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef))
        methods = {
            "is_valid_dtypes",
            "check_supported_dtypes",
            "is_valid_mma_tiler_and_cluster_shape",
            "check_mma_tiler_and_cluster_shape",
            "check_epilog_store_option",
            "is_valid_epilog_store_option",
        }
        cls.body = [x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name in methods]
        ns: dict = {"widths": DTYPES}
        module = ast.Module(
            body=[DenseSymbols().visit(copy.deepcopy(arch_guard)), DenseSymbols().visit(cls)],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(file), "exec"), ns)
        kind = ns[cls.name]
        obj = kind.__new__(kind)
        obj.use_2cta_instrs, obj.mma_tiler_mn, obj.cluster_shape_mn = False, (128, 128), (1, 1)
        obj.split_k, obj._mailbox_elems_per_thread = 1, 128
        obj.use_tma_store, obj.swizzle_size, obj.raster_along = False, 1, "m"
        cases = 0
        for accumulator, a, c in itertools.product(acc_inputs, DTYPES, DTYPES):
            obj.acc_dtype = accumulator
            expected = a in acc_inputs[accumulator] and c in acc_outputs[accumulator]
            if hasattr(obj, "is_valid_dtypes"):
                actual = obj.is_valid_dtypes(a, c)
            else:
                try:
                    obj.check_supported_dtypes(a, a, c)
                    actual = True
                except ValueError:
                    actual = False
            assert actual == expected, (file.name, accumulator, a, c)
            cases += 1
        obj.acc_dtype = "Float32"
        if hasattr(obj, "is_valid_mma_tiler_and_cluster_shape"):
            assert obj.is_valid_mma_tiler_and_cluster_shape()
        else:
            obj.check_mma_tiler_and_cluster_shape()
        if hasattr(obj, "is_valid_epilog_store_option"):
            assert obj.is_valid_epilog_store_option(128, 128)
            assert not obj.is_valid_epilog_store_option(120, 128)
        elif hasattr(obj, "check_epilog_store_option"):
            obj.check_epilog_store_option(128, 128)
            try:
                obj.check_epilog_store_option(120, 128)
            except ValueError:
                pass
            else:
                raise AssertionError("Direct tail should fail frontend")
        if file.name == "dense_gemm_persistent_dynamic.py":
            obj.split_k = 2
            obj.check_mma_tiler_and_cluster_shape()
            for attribute, value in (
                ("use_tma_store", True),
                ("swizzle_size", 2),
                ("raster_along", "n"),
            ):
                original = getattr(obj, attribute)
                setattr(obj, attribute, value)
                try:
                    obj.check_mma_tiler_and_cluster_shape()
                except ValueError:
                    pass
                else:
                    raise AssertionError("Invalid split-K policy was accepted")
                setattr(obj, attribute, original)
        # Keep the source target guard: the same Int8 guard rejects SM107 explicitly.
        try:
            ns["check_int8_mma_arch"]("Int8", "sm_107a")
        except ValueError:
            pass
        else:
            raise AssertionError("SM107 integer gate was dropped")
        counts[file.name] = cases
    pipelines = 0
    for stages, tiles in itertools.product(range(2, 10), range(33)):
        lookahead = min(stages - 2, tiles)
        produced, consumed, live = list(range(lookahead)), [], {}
        for tile in produced:
            live[tile % stages] = tile
        for tile in range(tiles):
            if tile < tiles - lookahead:
                future = tile + lookahead
                assert future % stages not in live
                live[future % stages] = future
                produced.append(future)
            assert live.pop(tile % stages) == tile
            consumed.append(tile)
        assert produced == consumed == list(range(tiles)) and not live
        pipelines += 1

    # Source alpha/beta arithmetic happens in epi_dtype before final D conversion.
    def half(x: float) -> float:
        return struct.unpack("<e", struct.pack("<e", x))[0]

    after_affine = half(2049.0 + -2048.0)
    before_affine = half(half(2049.0) + -2048.0)
    assert after_affine == 1.0 and before_affine == 0.0
    assert math.isnan(1.0 + 0.0 * math.nan)
    result = {
        "outcome": "PASS",
        "dtype_domains": counts,
        "manual_pipeline_domains": pipelines,
        "frontend_geometry_witness": {"mma_mn": [128, 128], "cluster_mn": [1, 1], "two_cta": False},
        "affine_cast_witness": {"convert_after": after_affine, "convert_before": before_affine},
        "beta_zero_nan_c": "NaN under explicit multiply/add; C is still loaded by source",
        "boundary": (
            "Source static predicates and index/numeric policies; no tensors/DSL compilation."
        ),
    }
    (STUDY / "checks/dense-contracts-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: six dtype/geometry menus, manual pipeline indices and affine cast witness.\n"
    )


if __name__ == "__main__":
    main()
