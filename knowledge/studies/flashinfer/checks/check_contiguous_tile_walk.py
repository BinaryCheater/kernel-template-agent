"""Logical flat-interval/group-segment witnesses; no persistent GPU execution."""

import json
import math
import struct
from pathlib import Path


def owner(s, workers, total):
    return ((s + 1) * workers - 1) // total


def bf16(x):
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


if __name__ == "__main__":
    cases = 0
    for groups in range(1, 9):
        for tiles in range(1, 10):
            total = groups * tiles
            for workers in range(1, total + 1):
                visited = []
                partials = {}
                counts = [
                    owner(g * tiles + tiles - 1, workers, total)
                    - owner(g * tiles, workers, total)
                    + 1
                    for g in range(groups)
                ]
                for c in range(workers):
                    lo, hi = c * total // workers, (c + 1) * total // workers
                    for s in range(lo, hi):
                        assert owner(s, workers, total) == c
                        visited.append(s)
                    for g in range(lo // tiles, (hi - 1) // tiles + 1):
                        a, b = max(0, lo - g * tiles), min(tiles, hi - g * tiles)
                        segment = c - owner(g * tiles, workers, total)
                        assert 0 <= segment < counts[g]
                        assert (g, segment) not in partials
                        # Unique powers distinguish a lost or repeated tile.
                        partials[g, segment] = sum(1 << t for t in range(a, b))
                assert visited == list(range(total))
                for g in range(groups):
                    assert sum(partials[g, s] for s in range(counts[g])) == (1 << tiles) - 1
                cases += 1
    assert owner(3, 3, 10) == 1 and 3 * 3 // 10 == 0
    groups, tiles, workers = 2, 5, 3
    intervals = [[c * 10 // 3, (c + 1) * 10 // 3] for c in range(3)]
    counts = [
        owner(g * tiles + tiles - 1, workers, 10) - owner(g * tiles, workers, 10) + 1
        for g in range(groups)
    ]
    assert counts == [2, 2]
    gate_weight = 1 / (1 + math.exp(-1.0))
    rounding_case = None
    for i in range(2048):
        x = 1 + i / 1024
        staged = bf16(bf16(x) * gate_weight)
        fused = bf16(x * gate_weight)
        if staged != fused:
            rounding_case = dict(
                attention=x, sigmoid_gate=gate_weight, cast_then_gate=staged, gate_then_cast=fused
            )
            break
    assert rounding_case is not None
    result = {
        "kind": "logical_contiguous_group_tile_walk",
        "native_execution": False,
        "command": (
            ".venv/bin/python knowledge/studies/flashinfer/checks/check_contiguous_tile_walk.py"
        ),
        "rectangle_worker_cases": cases,
        "example_intervals": intervals,
        "example_group_segment_counts": counts,
        "boundary_slot3_owner": 1,
        "wrong_floor_owner": 0,
        "gate_rounding_order_witness": rounding_case,
        "remaining": (
            "Native accumulator/group reset, persistent ring phases, partial s"
            "tate/LSE encoding and post-grid combine publication"
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
