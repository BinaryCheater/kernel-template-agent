"""Logical ordered FP32 merge, routed metadata and quantizer-policy witnesses."""

import itertools
import json
import math
import struct
from pathlib import Path


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bf16(x):
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]


if __name__ == "__main__":
    range_cases = 0
    for steps in range(2, 13):
        for splits in range(2, 10):
            ranges = [(s * steps // splits, (s + 1) * steps // splits) for s in range(splits)]
            valid = [s for s, (lo, hi) in enumerate(ranges) if hi > lo]
            assert len(valid) >= 2
            assert [i for lo, hi in ranges for i in range(lo, hi)] == list(range(steps))
            workspace = [float("nan")] * splits
            for s in valid:
                lo, hi = ranges[s]
                workspace[s] = sum(range(lo + 1, hi + 1))
            acc = 0.0
            for s in range(splits):
                lo, hi = ranges[s]
                if hi > lo:
                    acc = f32(acc + workspace[s])
            assert acc == steps * (steps + 1) // 2
            range_cases += 1
    valid_holes = [s for s in range(8) if (s + 1) * 2 // 8 > s * 2 // 8]
    assert valid_holes == [3, 7]
    partials = [2**24, 1.0, -(2**24)]
    results = []
    arrival_results = set()
    for order in itertools.permutations(range(3)):
        workspace = [None] * 3
        arrival = 0.0
        for s in order:
            workspace[s] = partials[s]
            arrival = f32(arrival + partials[s])
        acc = 0.0
        for value in workspace:
            acc = f32(acc + value)
        results.append(acc)
        arrival_results.add(arrival)
    assert set(results) == {0.0} and arrival_results == {0.0, 1.0}
    # Intermediate BF16 reductions lose small contributions that FP32 retains.
    pieces = [1.0] + [2**-9] * 4
    low = 0.0
    high = 0.0
    for piece in pieces:
        low = bf16(low + piece)
        high = f32(high + piece)
    assert bf16(high) != low
    # Route values and their per-row scale together; duplicate reads are real.
    route = [2, 0, 2]
    codes, scales = [2.0, 3.0, 5.0], [0.25, 0.5, 2.0]
    decoded = [codes[r] * scales[r] for r in route]
    assert decoded == [10.0, 0.5, 10.0]
    wrong = [codes[r] * scales[i] for i, r in enumerate(route)]
    assert wrong != decoded
    assert len(decoded) == 3 and len(set(route)) == 2
    # MX floor-anchor policy is not ceil(amax / max_code).
    amax = 500.0
    floor_scale = 2 ** math.floor(math.log2(amax)) / 256
    ceiling_scale = 2 ** math.ceil(math.log2(amax / 448))
    assert floor_scale == 1 and ceiling_scale == 2
    assert amax / floor_scale > 448
    # A full M32 quantizer cannot use one M16 half's maximum.
    half_max = [1.0, 16.0]
    assert max(half_max) != half_max[0]
    result = {
        "kind": "logical_ordered_workspace_route_check",
        "native_execution": False,
        "status": "passed",
        "range_cases": range_cases,
        "producer_completion_permutations": len(results),
        "scattered_live_slots": valid_holes,
        "arrival_order_results": sorted(arrival_results),
        "checks": [
            "floor_partition_cover",
            "inactive_holes_not_loaded",
            "fixed_slot_sum",
            "completion_order_independence",
            "fp32_until_final_cast",
            "payload_scale_same_route",
            "duplicate_route_reads",
            "mx_floor_vs_ceiling",
            "complete_quant_group",
        ],
        "numeric_boundary": (
            "Python/FP32/BF16 witnesses; native GEMM instruction association, "
            "route gathers, quantizer saturation and stream publication remain"
            " separate."
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
