"Compare actual block-scaled variant predicates on symbolic dtype identities."

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import sys
from pathlib import Path

from frontend_contract import DTYPES, Symbols

STUDY = Path(__file__).resolve().parents[1]
PREFIX = "examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/"


def extract(path: Path, manifest: dict) -> type:
    relative = str(path.relative_to(Path(manifest["tree"])))
    entry = next(x for x in manifest["files"] if x["path"] == relative)
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    cls = next(x for x in ast.parse(data).body if isinstance(x, ast.ClassDef))
    names = {
        "is_valid_dtypes_and_scale_factor_vec_size",
        "is_valid_layouts",
        "is_valid_mma_tiler_and_cluster_shape",
        "is_valid_tensor_alignment",
        "can_implement",
        "needs_unpack_tma",
    }
    cls.body = [x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name in names]
    module = ast.Module(body=[Symbols().visit(cls)], type_ignores=[])
    ns: dict = {"widths": DTYPES}
    exec(compile(ast.fix_missing_locations(module), relative, "exec"), ns)
    return ns[cls.name]


def main() -> None:
    # Extend symbolic identities only for this process; no DSL substitute is imported.
    DTYPES.update({"Float6E2M3FN": 6, "Float6E3M2FN": 6})
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"]) / PREFIX
    base = extract(root / "dense_blockscaled_gemm_persistent.py", manifest)
    pf = extract(root / "dense_blockscaled_gemm_persistent_prefetch.py", manifest)
    amax = extract(root / "dense_blockscaled_gemm_persistent_amax.py", manifest)
    sm103 = extract(root / "sm103_dense_blockscaled_gemm_persistent.py", manifest)
    outputs = ("Float32", "Float16", "BFloat16", "Float8E4M3FN", "Float8E5M2")
    operands = ("Float4E2M1FN", "Float6E2M3FN", "Float6E3M2FN", "Float8E4M3FN", "Float8E5M2")
    domains = 0
    for a, b, sf, vector, c in itertools.product(
        operands, operands, ("Float8E8M0FNU", "Float8E4M3FN"), (8, 16, 32), outputs
    ):
        expected = (vector == 16 and a == b == "Float4E2M1FN") or (
            vector == 32 and sf == "Float8E8M0FNU"
        )
        assert base.is_valid_dtypes_and_scale_factor_vec_size(a, b, sf, vector, c) == expected
        domains += 1
    unpack = 0
    for a, b in itertools.product(operands, repeat=2):
        expected = DTYPES[a] != DTYPES[b] or DTYPES[a] == 6 or DTYPES[b] == 6
        assert base.needs_unpack_tma(a, b) == expected
        unpack += 1
    for a, sf, vector, c in itertools.product(
        DTYPES, ("Float8E8M0FNU", "Float8E4M3FN"), (8, 16, 32), outputs
    ):
        assert pf.is_valid_dtypes_and_scale_factor_vec_size(a, sf, vector, c) == (
            amax.is_valid_dtypes_and_scale_factor_vec_size(a, sf, vector, c)
        )
    assert base.can_implement(
        (128, 256, 128, 1),
        "Float6E2M3FN",
        "Float8E4M3FN",
        "Float8E8M0FNU",
        "Float32",
        "k",
        "k",
        "n",
        32,
        (128, 128),
        (1, 1),
    )
    assert not base.is_valid_tensor_alignment(
        128, 256, 64, 1, "Float6E2M3FN", "Float8E4M3FN", "Float32", "k", "k", "n", (128, 128)
    )
    assert not base.is_valid_tensor_alignment(
        128, 256, 128, 1, "Float8E4M3FN", "Float6E2M3FN", "Float32", "k", "n", "n", (128, 192)
    )
    assert not base.is_valid_tensor_alignment(
        256, 256, 128, 1, "Float8E4M3FN", "Float6E2M3FN", "Float32", "k", "n", "n", (256, 128)
    )
    assert sm103.can_implement(
        "Float4E2M1FN",
        "Float8E8M0FNU",
        32,
        "Float16",
        (128, 128),
        (1, 1),
        128,
        128,
        768,
        1,
        "k",
        "k",
        "n",
        True,
    )
    assert not sm103.is_valid_mma_tiler_and_cluster_shape((128, 192), (1, 1))
    grouped_root = root.parent / "blockscaled_grouped_gemm"
    grouped100 = extract(grouped_root / "grouped_blockscaled_gemm.py", manifest)
    grouped103 = extract(grouped_root / "sm103_grouped_blockscaled_gemm.py", manifest)
    problems = [(128, 128, 768, 1), (256, 128, 0, 1), (128, 256, 1536, 1)]
    assert grouped100.can_implement(
        "Float4E2M1FN",
        "Float4E2M1FN",
        "Float8E8M0FNU",
        32,
        "Float16",
        (128, 128),
        (1, 1),
        problems,
        "k",
        "k",
        "n",
    )
    assert grouped103.can_implement(
        "Float4E2M1FN", "Float8E8M0FNU", 32, "Float16", (128, 128), (1, 1), problems, "k", "k", "n"
    )
    assert not grouped100.is_valid_dtypes_and_scale_factor_vec_size(
        "Float6E2M3FN", "Float8E4M3FN", "Float8E8M0FNU", 32, "Float16"
    )
    assert not grouped100.is_valid_dtypes_and_scale_factor_vec_size(
        "Float4E2M1FN", "Float4E2M1FN", "Float8E8M0FNU", 32, "Float4E2M1FN"
    )
    result = {
        "outcome": "PASS",
        "grouped_frontend_witness": problems,
        "grouped100_rejected_optional_paths": ["FP6 operand", "FP4 output"],
        "baseline_dtype_domains": domains,
        "unpack_pairs": unpack,
        "prefetch_amax_dtype_predicates": "equal on sampled symbols",
        "baseline_mixed_fp6_witness": {
            "mnkl": [128, 256, 128, 1],
            "mma_mn": [128, 128],
            "a": "Float6E2M3FN",
            "b": "Float8E4M3FN",
        },
        "rejected_neighbors": [
            "subbyte K64 unpack",
            "FP6 N-major tile192",
            "2CTA FP6 N-major tile128",
        ],
        "boundary": "Retained frontend predicates; no UNPACK instruction or device run.",
    }
    (STUDY / "checks/variant-contracts-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write("PASS: mixed/FP6 baseline, unpack alignment and narrower companion menus.\n")


if __name__ == "__main__":
    main()
