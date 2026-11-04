"""Replay blockwise scale ownership and cooperative output-prefix filtering."""

from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import sys
from fractions import Fraction
from pathlib import Path

from frontend_contract import DTYPES, Symbols

STUDY = Path(__file__).resolve().parents[1]
PREFIX = "examples/python/CuTeDSL/cute/blackwell/kernel/blockwise_gemm/"


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    entries = []
    predicates = 0
    for file in sorted((root / PREFIX).glob("*.py")):
        data = file.read_bytes()
        relative = file.relative_to(root).as_posix()
        retained = next(x for x in manifest["files"] if x["path"] == relative)
        assert hashlib.sha256(data).hexdigest() == retained["sha256"]
        tree = ast.parse(data)
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef))
        methods = {
            x.name: [x.lineno, x.end_lineno] for x in cls.body if isinstance(x, ast.FunctionDef)
        }
        entries.append({"path": relative, "sha256": retained["sha256"], "methods": methods})
        selected = copy.deepcopy(cls)
        selected.body = [
            x
            for x in selected.body
            if isinstance(x, ast.FunctionDef)
            and (x.name.startswith("is_valid_") or x.name == "can_implement")
        ]
        ns: dict = {"widths": DTYPES}
        module = ast.Module(body=[Symbols().visit(selected)], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), relative, "exec"), ns)
        kind = ns[cls.name]
        for ab, acc, c in itertools.product(DTYPES, repeat=3):
            expected = (
                ab in {"Float8E4M3FN", "Float8E5M2"}
                and acc == "Float32"
                and c in {"Float32", "Float16", "BFloat16"}
            )
            assert kind.is_valid_dtypes(ab, acc, c) == expected
            predicates += 1
        args = ["Float8E4M3FN", "Float32", "BFloat16", False, (128, 128), (1, 1)]
        assert kind.can_implement(*args, 256, 256, 256, 3, "k", "k", "n")
        assert not kind.can_implement(*args, 256, 256, 257, 3, "k", "k", "n")
        # The contiguous source rejects cluster footprints that cross128-row group alignment.
        wide = kind.is_valid_mma_tiler_and_cluster_shape(False, (128, 128), (2, 1))
        assert wide == (file.name != "contiguous_grouped_gemm.py")
        assignments = {
            ast.unparse(n.targets[0]): n.value.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.value, ast.Constant)
        }
        assert [assignments[f"self.scale_granularity_{x}"] for x in "mnk"] == [1, 128, 128]
        source = data.decode()
        assert "tiled_mma.set(tcgen05.Field.ACCUMULATE, False)" in source
        assert "tTR_rAcc_final.fill(0.0)" in source
        assert assignments["self.tmem_final_offset"] == 384

    # Independent decoded products vs admitted post-K128 scaling with unequal metadata.
    numeric_cases = 0
    wrong_final_scale = None
    for k_extent, m, n in itertools.product((0, 128, 144, 256, 384), (0, 1), (127, 128)):

        def qa(k: int, row: int = m) -> int:
            return (k + row) % 7 - 3

        def qb(k: int, column: int = n) -> int:
            return (2 * k + column) % 9 - 4

        def sa(t: int, row: int = m) -> Fraction:
            return Fraction((row + 1) * (t + 1), 2)

        def sb(t: int, column: int = n) -> Fraction:
            return Fraction((column // 128 + 2) * (t + 2), 4)

        decoded = sum(qa(k) * sa(k // 128) * qb(k) * sb(k // 128) for k in range(k_extent))
        promoted = sum(
            sum(qa(k) * qb(k) for k in range(t, min(t + 128, k_extent)))
            * sa(t // 128)
            * sb(t // 128)
            for t in range(0, k_extent, 128)
        )
        assert promoted == decoded
        if k_extent == 256 and m == 0 and n == 127:
            wrong = sum(qa(k) * qb(k) for k in range(k_extent)) * sa(1) * sb(1)
            assert wrong != decoded
            wrong_final_scale = {"correct": str(decoded), "one_final_scale": str(wrong)}
        numeric_cases += 1

    # Prefix filtering is per cooperative unit. Padded output rows are unobserved.
    work_domains = 0
    for unit, peers, capacity_units, workers, bypass in itertools.product(
        (64, 128, 256), (1, 2, 4), (1, 2, 3), (1, 2, 5), (False, True)
    ):
        assert unit % peers == 0
        capacity = unit * capacity_units
        for valid in sorted({0, 1, unit - 1, unit, min(capacity, unit + 1), capacity}):
            visible = []
            for worker in range(workers):
                for i, linear in enumerate(range(worker, capacity_units, workers)):
                    start = linear * unit
                    keep = start < valid or (bypass and i == 0)
                    if keep:
                        visible.extend(range(start, min(start + unit, valid)))
                    else:
                        assert start >= valid
            assert sorted(visible) == list(range(valid))
            work_domains += 1
    peer_local = [peer * 64 < 1 for peer in range(2)]
    assert peer_local == [True, False]  # wrong: two peers disagree on one128-row work unit

    # A row lookup at a tile start is valid only when no unit straddles group boundaries.
    lengths = (128, 256, 0, 128)
    row_group = [g for g, length in enumerate(lengths) for _ in range(length)]
    for unit in (64, 128):
        for start in range(0, len(row_group), unit):
            assert len(set(row_group[start : start + unit])) == 1
    bad_group_rows = [0] * 64 + [1] * 128
    assert bad_group_rows[0] != bad_group_rows[127]

    # Source advances then reads gidx_mapping even on invalid terminal work.
    source = (root / PREFIX / "masked_grouped_gemm.py").read_text()
    assert "cur_boundary = gidx_mapping[cur_tile_coord[2]]" in source
    total_clusters, groups = 6, 3  # two M units, one N unit per group
    terminal_batch = (total_clusters // 2) // 1
    assert terminal_batch == groups
    terminal_oob = False
    try:
        _ = [1, 128, 129][terminal_batch]
    except IndexError:
        terminal_oob = True
    assert terminal_oob
    # Raw TMEM stages occupy384 columns; the active final region starts there.
    for cta_rows, raw_stages in ((128, 3), (64, 6)):
        assert cta_rows * raw_stages == 384
        assert 384 + cta_rows <= 512
    result = {
        "outcome": "PASS",
        "dtype_predicates": predicates,
        "exact_scale_cases": numeric_cases,
        "wrong_final_scale": wrong_final_scale,
        "cooperative_filter_domains": work_domains,
        "peer_local_disagreement": peer_local,
        "aligned_row_groups": list(lengths),
        "terminal_metadata_index": terminal_batch,
        "metadata_groups": groups,
        "source_terminal_guard_gap": True,
        "boundary": (
            "Symbolic frontend/source equations and rational arithmetic; no FP"
            "32 error or device progress claim."
        ),
    }
    (STUDY / "inputs/blockwise-variants.json").write_text(
        json.dumps(
            {
                "entries": entries,
                "witness": (
                    "FP8 E4M3, FP32 raw/final, BF16 C, MMA128x128, cluster1x1, M256/N256/K256/L3"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    (STUDY / "checks/blockwise-contracts-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    sys.stdout.write(
        "PASS: blockwise scale ownership, aligned row maps and cooperative"
        " prefix work; terminal guard gap retained.\n"
    )


if __name__ == "__main__":
    main()
