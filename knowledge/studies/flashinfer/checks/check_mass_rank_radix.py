"Exact weighted radix crossings, whole cohorts and bounded candidate rescans."

import itertools
import json
import struct
from collections import Counter
from fractions import Fraction as F
from pathlib import Path


def key(value, width):
    bits = int.from_bytes(struct.pack("<f" if width == 32 else "<e", float(value)), "little")
    return (~(bits ^ (1 << (width - 1)))) & ((1 << width) - 1)


def radix(values, fraction, width, capacity, compact):
    encoded = [(key(v, width), v) for v in values]
    residual = fraction * sum(values)
    assert residual > 0
    previous_shift = width
    prefix = 0
    cache = None
    cached_passes = 0
    fallback_passes = 0
    for p in range((width + 10) // 11):
        shift = max(width - 11 * (p + 1), 0)
        mask = (1 << (previous_shift - shift)) - 1
        active = [
            (k, v)
            for k, v in (cache if cache is not None else encoded)
            if (k >> previous_shift) << previous_shift == prefix
        ]
        expected = [(k, v) for k, v in encoded if (k >> previous_shift) << previous_shift == prefix]
        assert Counter(active) == Counter(expected)
        if compact and p > 0 and len(active) <= capacity:
            next_cache = active[:]
            cached_passes += 1
            assert len(next_cache) <= capacity
        else:
            next_cache = None
            fallback_passes += int(compact and p > 0)
        mass, count = {}, {}
        for k, v in active:
            bucket = (k >> shift) & mask
            mass[bucket] = mass.get(bucket, F(0)) + v
            count[bucket] = count.get(bucket, 0) + 1
        earlier = F(0)
        selected = None
        for bucket in sorted(count):
            if earlier + mass[bucket] >= residual:
                selected = bucket
                break
            earlier += mass[bucket]
        assert selected is not None and mass[selected] > 0
        residual -= earlier
        assert 0 < residual <= mass[selected]
        prefix |= selected << shift
        cache, previous_shift = next_cache, shift
    threshold = next(v for k, v in encoded if k == prefix)
    retained = sum(v for v in values if v >= threshold)
    result = tuple(v / retained if v >= threshold else F(0) for v in values)
    return threshold, result, cached_passes, fallback_passes


def verify(values, fraction, width, capacity, compact):
    mass = F(0)
    for threshold in sorted(values, reverse=True):
        mass += threshold
        if mass >= fraction * sum(values):
            break
    retained = sum(v for v in values if v >= threshold)
    expected = tuple(v / retained if v >= threshold else F(0) for v in values)
    actual, result, cached, fallback = radix(values, fraction, width, capacity, compact)
    assert actual == threshold and result == expected and sum(result) == 1
    return cached, fallback


if __name__ == "__main__":
    cases = cached_passes = fallback_passes = 0
    for n in range(1, 5):
        for integers in itertools.product(range(4), repeat=n):
            if sum(integers) == 0:
                continue
            values = tuple(F(v) for v in integers)
            for fraction in [F(1, 4), F(1, 2), F(3, 4), F(1)]:
                for width in [16, 32]:
                    for compact, capacity in [
                        (False, 0),
                        (True, 0),
                        (True, 1),
                        (True, 2),
                        (True, 4),
                    ]:
                        cached, fallback = verify(values, fraction, width, capacity, compact)
                        cached_passes += cached
                        fallback_passes += fallback
                        cases += 1
    # Differences in low mantissa bits require the later radix digits.
    low_bit_cases = 0
    for width, mantissa_bits in [(16, 10), (32, 23)]:
        values = (F(1), F(1) + F(1, 2**mantissa_bits), F(1) + F(2, 2**mantissa_bits), F(9, 8))
        for fraction in [F(1, 4), F(1, 2), F(3, 4), F(1)]:
            for capacity in [0, 1, 2, 4]:
                verify(values, fraction, width, capacity, True)
                low_bit_cases += 1
    threshold, result, _, _ = radix((F(5, 8), F(1, 8), F(1, 8), F(1, 8)), F(3, 4), 32, 1, True)
    # Exactly representable masses retain the complete equality cohort.
    assert threshold == F(1, 8) and result == (F(5, 8), F(1, 8), F(1, 8), F(1, 8))
    weighted, _, _, _ = radix((F(8), F(1), F(1)), F(1, 2), 32, 1, True)
    assert weighted == 8 and sorted([8, 1, 1], reverse=True)[1] == 1
    report = {
        "kind": "exact_mass_rank_radix_boundary",
        "status": "passed",
        "native_execution": False,
        "reference_cases": cases,
        "low_mantissa_cases": low_bit_cases,
        "logical_cached_passes": cached_passes,
        "logical_rescan_passes": fallback_passes,
        "schedules": {"16": [11, 5], "32": [11, 11, 10]},
        "capacity_options": [0, 1, 2, 4],
        "weighted_vs_count_boundary": {"values": [8, 1, 1], "mass_half": 8, "count_second": 1},
        "whole_boundary_cohort": {"threshold": str(threshold), "retained_mass": "1"},
        "boundary": "Exact mass oracle; native sum reconstruction/publication not executed",
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
