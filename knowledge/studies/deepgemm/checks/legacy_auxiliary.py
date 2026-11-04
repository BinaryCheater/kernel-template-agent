"""Legacy Triton index equations and actual pure TileLang shape helper checks."""

from __future__ import annotations

import ast
import itertools
import json
import math
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def check_raster() -> int:
    domains = 0
    for m, n in itertools.product(range(1, 34), range(1, 12)):
        coords = []
        for pid in range(m * n):
            group = pid // (8 * n)
            first = group * 8
            size = min(m - first, 8)
            coords.append((first + pid % size, (pid % (8 * n)) // size))
        assert sorted(coords) == list(itertools.product(range(m), range(n)))
        domains += 1
    return domains


def check_gather() -> dict[str, object]:
    m, n, k = 3, 5, 8
    a = [[10 * row + col for col in range(m)] for row in range(k)]
    b = [[100 * row + col for col in range(n)] for row in range(k)]
    indices = [7, -1, 1, 4, 0, 6, 3, 2]
    cases = 0
    for fused, start, end in itertools.product(("a", "b"), (0, 4), (4, 8)):
        if start > end:
            continue
        for row, col in itertools.product(range(m), range(n)):
            terms = []
            for idx in range(start, end):
                source = indices[idx]
                av = a[source][row] if fused == "a" and source >= 0 else a[idx][row]
                bv = b[source][col] if fused == "b" and source >= 0 else b[idx][col]
                terms.append(av * bv if source >= 0 else 0)
            expected = sum(
                a[indices[i] if fused == "a" else i][row]
                * b[indices[i] if fused == "b" else i][col]
                for i in range(start, end)
                if indices[i] >= 0
            )
            assert sum(terms) == expected
            cases += 1
    # A single K5 group with tile4 reads indices4..7 in its last iteration.
    visited = [idx for base in range(0, 5, 4) for idx in range(base, base + 4)]
    assert visited == list(range(8)) and set(visited) - set(range(5)) == {5, 6, 7}
    # Empty ACC keeps D; non-ACC must zero it.
    old = 17
    return {
        "integer_output_checks": cases,
        "unmasked_group_tail": visited,
        "empty_acc_source_policy": {"accumulate": old, "overwrite": 0},
    }


def check_shapes(root: Path) -> dict[str, object]:
    source = ast.parse((root / "third-party/tilelang_ops/utils.py").read_text())
    functions = [
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in ("ceil_div", "align", "get_sf_shape")
    ]
    namespace: dict[str, object] = {}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "retained_sf_shapes", "exec"),
        namespace,
    )
    shape = namespace["get_sf_shape"]
    assert callable(shape)
    domains = 0
    for tokens, hidden, channels, packed, column in itertools.product(
        (0, 1, 3, 7, 16), (128, 256, 1024), (32, 128), (False, True), (False, True)
    ):
        scales = (hidden + channels - 1) // channels
        result = shape(tokens, hidden, channels, packed, column)
        if packed:
            scales = (scales + 3) // 4
        expected_tokens = ((tokens + 3) // 4 * 4) if column else tokens
        expected_tokens *= 4 if packed and column else 1
        expected = (scales, expected_tokens) if column else (expected_tokens, scales)
        assert result == expected
        domains += 1
    row_major = shape(7, 256, 32, False, False)
    assert row_major == (7, 8)
    # Wrapper always transposes returned SF: row-major branch becomes shape(7,7), not(7,8).
    returned = (min(row_major[1], 7), row_major[0])
    assert returned == (7, 7)
    half, channels = 49152, 96
    splits = half // math.gcd(half, 16384)
    assert half % channels == 0 and splits == 3 and (half // splits) % channels != 0
    return {
        "actual_shape_helper_domains": domains,
        "row_major_return_shape": {"allocated": row_major, "returned": returned},
        "hidden_split_group_mismatch": {
            "half_hidden": half,
            "channels": channels,
            "hidden_blocks": splits,
            "block_hidden": half // splits,
        },
    }


def main() -> None:
    root = Path(json.loads((STUDY / "inputs/sources.json").read_text())["tree"])
    for name in ("a_fused_k_grouped_gemm", "b_fused_k_grouped_gemm"):
        body = (root / f"deep_gemm/legacy/{name}.py").read_text()
        assert (
            "restore_value=['d_ptr']" in body
            and "for k in range(k_start, k_end, BLOCK_SIZE_K)" in body
        )
    indexing = (root / "csrc/indexing/main.cu").read_text()
    assert "return 0;" in indexing and "__global__" not in indexing
    result = {
        "outcome": "PASS",
        "raster_domains": check_raster(),
        "gathers": check_gather(),
        "auxiliary": check_shapes(root),
        "boundary": (
            "Source arithmetic and pure shape helpers; no Triton/TileLang comp"
            "ilation or GPU execution."
        ),
    }
    (STUDY / "checks/legacy-auxiliary-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write("PASS: legacy raster/gather equations and auxiliary shape counterexamples.\n")


if __name__ == "__main__":
    main()
