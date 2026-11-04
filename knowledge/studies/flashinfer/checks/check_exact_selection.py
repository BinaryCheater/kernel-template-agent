"Logical threshold certificate and radix boundary witnesses, not CUDA execution."

import itertools
import json
import math
import struct
from collections import Counter
from pathlib import Path


def order_key(value, fold_zero=True):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if fold_zero and bits & 0x7FFFFFFF == 0:
        bits = 0
    return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else bits | 0x80000000


def kth_key(keys, want, schedule):
    assert 1 <= want <= len(keys) and sum(schedule) == 32
    active = list(keys)
    shift = 32
    for width in schedule:
        shift -= width
        bins = Counter((key >> shift) & ((1 << width) - 1) for key in active)
        before = 0
        for digit, count in sorted(bins.items(), reverse=True):
            if before + count >= want:
                want -= before
                active = [key for key in active if (key >> shift) & ((1 << width) - 1) == digit]
                break
            before += count
        assert 1 <= want <= len(active)
    assert len(set(active)) == 1
    return active[0], want, len(active)


def reference(records, k):
    return sorted(records, key=lambda r: (r[0], r[1]), reverse=True)[:k]


def radix_topk(records, k, schedule):
    want = min(k, len(records))
    if want == 0:
        return []
    boundary, ties, eq = kth_key([order_key(s) for s, _ in records], want, schedule)
    tied = [i for s, i in records if order_key(s) == boundary]
    assert len(tied) == eq
    cut, _, _ = kth_key(tied, ties, schedule)
    return sorted(
        (s, i)
        for s, i in records
        if order_key(s) > boundary or (order_key(s) == boundary and i >= cut)
    )


def filter_topk(records, k, capacity, threshold):
    qualifiers = [r for r in records if r[0] >= threshold]
    count = len(qualifiers)
    buffer = qualifiers[:capacity]
    want = min(k, len(records))
    accepted = want <= count <= capacity
    return reference(buffer if accepted else records, want), accepted, count


if __name__ == "__main__":
    cases, accepted_cases, fallback_cases = 0, 0, 0
    for n in range(6):
        for values in itertools.product([-1.0, 0.0, 2.0], repeat=n):
            records = [(s, i) for i, s in enumerate(values)]
            for k in [1, 2, 4]:
                for capacity in [k, k + 2]:
                    for threshold in [-math.inf, -1.0, 0.0, 1.0, 2.0, math.inf]:
                        out, accepted, _ = filter_topk(records, k, capacity, threshold)
                        assert out == reference(records, k)
                        accepted_cases += accepted
                        fallback_cases += not accepted
                        cases += 1
    overflow = [(float(i), i) for i in range(7)]
    out, accepted, count = filter_topk(overflow, 2, 3, -1.0)
    assert not accepted and count == 7 and out == reference(overflow, 2)
    assert reference(overflow[:3], 2) != out
    underfull, accepted, count = filter_topk(overflow, 2, 3, 6.0)
    assert not accepted and count == 1 and underfull == reference(overflow, 2)

    schedules = [(8, 8, 8, 8), (11, 11, 10)]
    records = [(s, i) for i, s in enumerate([-4.0, -0.0, 0.0, 3.0, 3.0, 1.5, -2.0, 3.0, 0.0, -0.0])]
    radix_cases = 0
    for schedule in schedules:
        for perm in [records, list(reversed(records)), records[::2] + records[1::2]]:
            for k in range(1, len(records) + 3):
                actual = radix_topk(perm, k, schedule)
                assert sorted(i for _, i in actual) == sorted(i for _, i in reference(records, k))
                radix_cases += 1
        # Crowded equal boundary cannot be truncated before the larger-ID cut.
        crowded = [(1.0, i) for i in range(41)] + [(2.0, 41)]
        picked = radix_topk(crowded, 5, schedule)
        assert sorted(i for _, i in picked) == [37, 38, 39, 40, 41]
    zeros = [(-0.0, 7), (0.0, 1)]
    assert radix_topk(zeros, 1, schedules[0])[0][1] == 7
    assert max(zeros, key=lambda r: (order_key(r[0], fold_zero=False), r[1]))[1] == 1
    result = {
        "kind": "logical_count_certified_filter_and_radix_boundary",
        "native_execution": False,
        "command": (
            ".venv/bin/python knowledge/studies/flashinfer/checks/check_exact_selection.py"
        ),
        "threshold_cases": cases,
        "accepted_cases": accepted_cases,
        "fallback_cases": fallback_cases,
        "overflow_full_count": 7,
        "truncated_overflow_selection_differs": True,
        "underfull_rescan_recovers": True,
        "radix_cases": radix_cases,
        "digit_schedules": schedules,
        "crowded_tie_winners": [37, 38, 39, 40, 41],
        "signed_zero_fold_required_for_larger_id_tie": True,
        "remaining": (
            "Native score-bit consistency, append/histogram atomic visibility,"
            " last-arriver progress, scratch generations and distributed emiss"
            "ion"
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
