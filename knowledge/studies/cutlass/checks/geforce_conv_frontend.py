"Replay retained SM120 convolution predicates and descriptor coordinate parameters."

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import yaml
from dense_contracts import DenseSymbols
from frontend_contract import DTYPES

from kta.core.expr import Expr

STUDY = Path(__file__).resolve().parents[1]
PREFIX = "examples/python/CuTeDSL/cute/blackwell_geforce/kernel/conv/"


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    entries, bodies = [], []
    names = {
        "_check_tensor_alignment",
        "_check_im2col_descriptor_limits",
        "_compute_im2col_params",
        "compute_zpq",
        "sf_k_tile_channels",
        "sfb_per_position_channels",
    }
    classes = []
    for filename in ("dense_implicit_gemm_fprop.py", "dense_blockscaled_implicit_gemm_fprop.py"):
        path = PREFIX + filename
        data = (root / path).read_bytes()
        retained = next(x for x in manifest["files"] if x["path"] == path)
        assert hashlib.sha256(data).hexdigest() == retained["sha256"]
        tree = ast.parse(data)
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef))
        entries.append(
            {
                "path": path,
                "sha256": retained["sha256"],
                "functions": {
                    x.name: [x.lineno, x.end_lineno]
                    for x in tree.body
                    if isinstance(x, ast.FunctionDef)
                },
                "methods": {
                    x.name: [x.lineno, x.end_lineno]
                    for x in cls.body
                    if isinstance(x, ast.FunctionDef)
                },
                "status": (
                    "Source-extracted across four convolution traces; native layout/nu"
                    "meric/resource/address/CLC progress obligations retained"
                ),
            }
        )
        bodies.extend(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names)
        cls.body = [
            x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "can_implement"
        ]
        classes.append(cls)
    ns: dict = {"widths": DTYPES}
    selected = ast.Module(body=[*bodies, *classes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(DenseSymbols().visit(selected)), PREFIX, "exec"), ns)
    dense_kind, scaled_kind = (ns[x.name] for x in classes)
    dense = dense_kind.__new__(dense_kind)
    dense.atom_layout, dense.cluster_shape_mnk = (2, 2, 1), (1, 1, 1)
    geometry = ((1, 1, 1),) * 3 + ((0, 0, 0),) * 2

    def accepts(fn, *args) -> bool:
        try:
            fn(*args)
        except ValueError:
            return False
        return True

    dense_cases = dense_accepted = 0
    for tm, tn, tk, ab, acc, out in itertools.product(
        (32, 64, 96, 128), (16, 32, 64), (16, 32, 64, 128), DTYPES, DTYPES, DTYPES
    ):
        dense.tile_shape_mnk = (tm, tn, tk)
        expected = (
            ab in {"Float16", "BFloat16", "Float8E4M3FN", "Float8E5M2"}
            and acc in {"Float16", "Float32"}
            and out in {"Float16", "BFloat16", "Float32", "Float8E4M3FN", "Float8E5M2"}
            and tm in {32, 64, 128}
            and tn in {32, 64}
            and tk in {32, 64, 128}
        )
        actual = accepts(dense.can_implement, 1, 128, 128, ab, acc, out, *geometry)
        assert actual == expected
        dense_cases += 1
        dense_accepted += actual
    scaled_cases = scaled_accepted = 0
    for tm, tn, tk, ab, sf, vec, c, k in itertools.product(
        (64, 128),
        (32, 64, 96, 128),
        (32, 64, 128),
        ("Float4E2M1FN", "Float8E4M3FN", "Float8E5M2"),
        ("Float8E4M3FN", "Float8E8M0FNU"),
        (16, 32),
        (32, 64, 96, 128, 192, 256),
        (64, 96, 128, 192),
    ):
        format_ok = (ab, sf, vec) in {
            ("Float4E2M1FN", "Float8E4M3FN", 16),
            ("Float8E4M3FN", "Float8E8M0FNU", 32),
        }
        expected = (
            format_ok
            and tm == 128
            and tn in {64, 96, 128}
            and tk in ((64,) if ab == "Float8E4M3FN" else (64, 128))
            and (tn != 96 or k <= 96)
            and c % (4 * vec) == 0
        )
        actual = accepts(
            scaled_kind.can_implement,
            (tm, tn, tk),
            ab,
            sf,
            "Float32",
            "BFloat16",
            vec,
            c,
            k,
            *geometry,
        )
        assert actual == expected
        scaled_cases += 1
        scaled_accepted += actual
    scaled_output_cases = 0
    for acc, out in itertools.product(DTYPES, DTYPES):
        assert accepts(
            scaled_kind.can_implement,
            (128, 64, 64),
            "Float4E2M1FN",
            "Float8E4M3FN",
            acc,
            out,
            16,
            64,
            128,
            *geometry,
        ) == (acc == "Float32" and out in {"Float16", "BFloat16"})
        scaled_output_cases += 1

    # Different argument order from the SM100 helper: do not share that call wrapper.
    descriptor_cases = 0
    for axis, flt, dil, up, lo, stride in itertools.product(
        range(3), (1, 2, 17, 33), (1, 2), (0, 15, 16), (0, 16, 17), (1, 8, 9)
    ):
        values = [[1] * 3, [1] * 3, [1] * 3, [0] * 3, [0] * 3]
        for seq, value in zip(values, (flt, stride, dil, up, lo), strict=True):
            seq[axis] = value
        expected = -16 <= -lo <= 15 and -16 <= up - (flt - 1) * dil <= 15
        expected &= (flt - 1) * dil <= 31 and stride <= 8
        assert accepts(ns["_check_im2col_descriptor_limits"], *map(tuple, values)) == expected
        descriptor_cases += 1
    params = ns["_compute_im2col_params"]((2, 3, 4), (1, 2, 3), (4, 5, 6), (2, 3, 4), (1, 2, 3))
    assert params == (
        (-6, -5, -4),
        (-6, -2, 0),
        (6, 5, 4),
        (3, 2, 1),
        (4, 3, 2),
        (0, 0, 0),
        (3, 2, 1),
    )
    # Distinct axes expose reversing DHW incorrectly and exchanging padding sides.
    assert params[0] != (-3, -2, -1)
    assert params[4] != (2, 3, 4)

    # The existing rule supplies stronger semantic premises than these predicates.
    dense.tile_shape_mnk = (128, 64, 64)
    witness_geometry = ((3,) * 3,) + ((1,) * 3,) * 4
    out = ns["compute_zpq"]((5,) * 3, (3,) * 3, *((1,) * 3,) * 4)
    assert out == (5,) * 3
    assert accepts(
        dense.can_implement,
        math.prod(out),
        128,
        128,
        "Float16",
        "Float32",
        "Float16",
        *witness_geometry,
    )
    assert accepts(
        scaled_kind.can_implement,
        (128, 64, 64),
        "Float4E2M1FN",
        "Float8E4M3FN",
        "Float32",
        "Float16",
        16,
        128,
        128,
        *witness_geometry,
    )
    assert accepts(dense.can_implement, 1, 0, 128, "Float16", "Float32", "Float16", *geometry)
    assert accepts(
        scaled_kind.can_implement,
        (128, 64, 64),
        "Float4E2M1FN",
        "Float8E4M3FN",
        "Float32",
        "Float16",
        16,
        0,
        128,
        *geometry,
    )
    bad_out = ns["compute_zpq"]((1, 1, 1), (3, 3, 1), (1,) * 3, (0,) * 3, (0,) * 3, (1,) * 3)
    assert bad_out == (-1, -1, 1) and math.prod(bad_out) == 1
    assert accepts(
        dense.can_implement,
        math.prod(bad_out),
        64,
        128,
        "Float16",
        "Float32",
        "Float16",
        (3, 3, 1),
        (1,) * 3,
        (1,) * 3,
        (0,) * 3,
        (0,) * 3,
    )
    ns["_check_im2col_descriptor_limits"]((1,) * 3, (0,) * 3, (1,) * 3, (0,) * 3, (0,) * 3)
    span_cases = 0
    for c, tk, vec in itertools.product((64, 128, 192, 256, 384), (64, 128), (16, 32)):
        span = ns["sf_k_tile_channels"](tk, vec)
        padded = ns["sfb_per_position_channels"](c, tk, vec)
        assert span == max(tk, 4 * vec)
        assert padded >= c and padded % span == 0 and padded - c < span
        span_cases += 1
    assert ns["sfb_per_position_channels"](192, 64, 32) == 256
    rule = yaml.safe_load((STUDY / "delta/implicit_patch_contraction@1/template.yaml").read_text())
    premises = {
        "extents_positive": True,
        "strides_positive": True,
        "dilations_positive": True,
        "padding_nonnegative": True,
        "output_extents_match": True,
        "input_filter_metadata_immutable": True,
        "zero_fill_matches_contract": True,
        "matrix_arithmetic_admitted": True,
        "patch_mapping_supported": True,
        "tail_accesses_safe": True,
        "output_mapping_exact": True,
    }
    scope = Expr.parse(rule["scope"]["predicate"], {"patch_storage"})
    assert scope.eval({"patch_storage": "on_load"}, p=premises)
    for name in premises:
        assert not scope.eval({"patch_storage": "on_load"}, p={**premises, name: False})
    result = {
        "outcome": "PASS",
        "dense_predicate_cases": dense_cases,
        "dense_accepted": dense_accepted,
        "scaled_predicate_cases": scaled_cases,
        "scaled_accepted": scaled_accepted,
        "scaled_output_cases": scaled_output_cases,
        "descriptor_cases": descriptor_cases,
        "scale_span_cases": span_cases,
        "existing_rule_scope_premise_mutations": len(premises),
        "distinct_axis_descriptor_parameters": params,
        "accepted_semantic_counterexamples": {
            "C_zero": ["dense", "scaled"],
            "dense_negative_output_axes": list(bad_out),
            "descriptor_zero_stride": True,
        },
        "boundary": (
            "Retained pure predicates/helper arithmetic with symbolic dtype id"
            "entities; no native compile, layout or execution proof."
        ),
    }
    (STUDY / "inputs/geforce-conv-variants.json").write_text(
        json.dumps(
            {
                "entries": entries,
                "active_slice": "Two-file family source-extracted; GeForce dense GEMM next",
            },
            indent=2,
        )
        + "\n"
    )
    (STUDY / "checks/geforce-conv-frontend-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    sys.stdout.write("PASS: SM120 convolution frontend menus, descriptor axes and scale spans.\n")


if __name__ == "__main__":
    main()
