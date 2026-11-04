"""Verify packed frame geometry and expose the cross-slot lifetime premise."""

import json
from pathlib import Path


def region(slot, phase, large, small):
    start = slot * (large + small) // 2
    if slot == 1:
        start += (large - small) // 2 * (1 - 2 * phase)
    size = large if (slot + phase) % 2 == 0 else small
    return start, start + size


def overlap(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])


if __name__ == "__main__":
    cases = []
    for large, small in [(12, 8), (49152, 32768), (16, 16)]:
        frames = [[region(i, phase, large, small) for i in range(3)] for phase in range(2)]
        for frame in frames:
            assert all(0 <= start < end <= 2 * large + small for start, end in frame)
            assert all(not overlap(frame[i], frame[j]) for i in range(3) for j in range(i))
        counterexample = overlap(frames[0][0], frames[1][1])
        assert counterexample == (large > small)
        cases.append(
            {
                "large_bytes": large,
                "small_bytes": small,
                "frames": frames,
                "capacity_bytes": 2 * large + small,
                "geometry": "passed",
                "same_slot_release_alone_insufficient": counterexample,
            }
        )
    Path(__file__).with_suffix(".json").write_text(
        json.dumps(
            {
                "kind": "logical_storage_geometry_and_lifetime_counterexample",
                "native_execution": False,
                "cases": cases,
                "remaining": (
                    "Source native phase, padded footprint and async reader retirement binding"
                ),
            },
            indent=2,
        )
        + "\n"
    )
