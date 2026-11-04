"""Replay scheduler address arithmetic against independent group enumerations."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def align(value: int, divisor: int) -> int:
    return ceil_div(value, divisor) * divisor


def swizzle(
    index: int, ms: int, ns: int, width: int, on_a: bool, multicast: int
) -> tuple[int, int]:
    primary, secondary = (ns, ms) if on_a else (ms, ns)
    group, within = divmod(index, secondary * width)
    first = group * width
    count = min(width, primary - first)
    if multicast > 1 and count % 2:
        even = count ^ 1
        if within < even * secondary:
            count = even
        else:
            within -= even * secondary
            first += even
            count = 1
    if on_a:
        return within // count, first + within % count
    return first + within % count, within // count


def masked_schedule(
    lengths: tuple[int, ...], ntiles: int, grid: int, block: int
) -> list[tuple[int, int, int]]:
    scheduled = []
    for cta in range(grid):
        group, prefix, iteration = 0, 0, 0
        while True:
            task = iteration * grid + cta
            while group < len(lengths):
                rows = ceil_div(lengths[group], block)
                if task < (prefix + rows) * ntiles:
                    break
                prefix += rows
                group += 1
            if group == len(lengths):
                break
            m, n = swizzle(task - prefix * ntiles, rows, ntiles, 8, True, 1)
            scheduled.append((group, m, n))
            iteration += 1
    return scheduled


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    source = (Path(manifest["tree"]) / "deep_gemm/include/deep_gemm/scheduler/gemm.cuh").read_text()
    for marker in (
        "(++ current_iter) * kNumSMs + blockIdx.x",
        "current_m_cumsum + num_m_blocks",
        "current_sf_k_cumsum += math::ceil_div(aligned_shape_k, kSFKSpan)",
        "current_k_start = math::align(current_k_end, kKAlignment)",
        "current_psum_m - last_psum_m",
        "num_blocks_in_group ^ 1",
    ):
        assert marker in source
    domains = 0
    for ms, ns, width, on_a, multicast in itertools.product(
        (1, 2, 7, 8, 9, 17), (1, 2, 7, 8, 9), (8, 16), (False, True), (1, 2)
    ):
        outputs = [swizzle(i, ms, ns, width, on_a, multicast) for i in range(ms * ns)]
        assert len(set(outputs)) == ms * ns
        assert set(outputs) == set(itertools.product(range(ms), range(ns)))
        domains += 1
    masked_cases = 0
    for lengths in itertools.product((0, 1, 63, 64, 65, 128), repeat=3):
        for grid in (1, 2, 7, 132):
            actual = masked_schedule(lengths, 2, grid, 64)
            expected = {
                (g, m, n)
                for g, length in enumerate(lengths)
                for m in range(ceil_div(length, 64))
                for n in range(2)
            }
            assert len(actual) == len(expected) and set(actual) == expected
            masked_cases += 1
    witness = masked_schedule((65, 0, 1), 2, 132, 64)
    # Output tensor-map capacity is 128; stores are full tiles, not a row mask.
    writes = {
        g: sorted(
            {
                row
                for gg, m, _ in witness
                if gg == g
                for row in range(m * 64, min((m + 1) * 64, 128))
            }
        )
        for g in range(3)
    }
    assert writes == {0: list(range(128)), 1: [], 2: list(range(64))}
    assert len(writes[0]) != 65 and len(writes[2]) != 1
    # Prefix ends: starts are aligned previous ends, not the previous ends themselves.
    ends = (65, 193, 321)
    starts = [0, *(align(end, 128) for end in ends[:-1])]
    assert starts == [0, 128, 256]
    live = [range(start, end) for start, end in zip(starts, ends, strict=True)]
    assert [len(rows) for rows in live] == [65, 65, 65]
    assert starts[1] != ends[0]
    # Independently packed K-group SF rows have their own prefix, not physical K/128.
    real_k = (1000, 4096, 900)
    physical_starts, sf_starts, end, sf_prefix = [], [], 0, 0
    for length in real_k:
        start = align(end, 128)
        physical_starts.append(start)
        sf_starts.append(sf_prefix)
        end = start + length
        sf_prefix += ceil_div(align(length, 128), 512)
    assert physical_starts == [0, 1024, 5120]
    assert sf_starts == [0, 2, 10]
    # At gran_k=32 each packed row spans 128 K; at gran_k=128 it spans 512.
    # Both are separate from the exact K end and independently rounded per group.
    result = {
        "outcome": "PASS",
        "swizzle_domains": domains,
        "masked_domains": masked_cases,
        "masked_witness": {
            "lengths": [65, 0, 1],
            "capacity": 128,
            "block_m": 64,
            "tile_count": len(witness),
            "written_row_counts": {str(k): len(v) for k, v in writes.items()},
        },
        "psum_m": {"ends": ends, "starts": starts},
        "k_grouped_gran128": {
            "lengths": real_k,
            "physical_starts": physical_starts,
            "packed_sf_starts": sf_starts,
        },
        "mutations_rejected": ["masked_length_equals_store_extent", "unrounded_prefix_start"],
        "boundary": (
            "Transcribed source index equations versus independent enumeration"
            "s; no compiled scheduler or GPU memory-safety check."
        ),
    }
    (STUDY / "checks/grouped-ownership-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        f"PASS: {domains} swizzle domains; {masked_cases} masked schedules; "
        "explicit M/K prefix witnesses."
    )


if __name__ == "__main__":
    main()
    sys.stdout.write("\n")
