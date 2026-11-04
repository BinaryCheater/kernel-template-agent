"Exact-unit fixed-point mass codec, integer scan and downstream conversion witnesses."

import itertools
import json
import random
import struct
from fractions import Fraction as F
from pathlib import Path


def fp32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def power2(exponent):
    return F(2**exponent) if exponent >= 0 else F(1, 2**-exponent)


def encode(values, padding):
    peak = max(values)
    e0 = max(bits(peak) >> 23, 1)
    unit = power2(e0 - 150 - padding)
    masses = []
    for value in values:
        raw = bits(value)
        exponent = raw >> 23
        significand = (raw & 0x7FFFFF) | (0x800000 if exponent else 0)
        shift = e0 - max(exponent, 1)
        mass = (significand << padding) >> shift if shift < 64 else 0
        assert mass == F(value) // unit
        assert 0 <= F(value) - mass * unit < unit
        masses.append(mass)
    assert masses and sum(masses) > 0
    assert max(masses) < 2 ** (24 + padding) and sum(masses) < 2**63
    return masses, unit


def check(values, padding):
    masses, unit = encode(values, padding)
    prefix = list(itertools.accumulate(masses))
    regrouped, before = [], 0
    for start in range(0, len(masses), 64):
        local = list(itertools.accumulate(masses[start : start + 64]))
        regrouped.extend(before + x for x in local)
        before += local[-1]
    assert prefix == regrouped
    weight = sum(map(F, values), F(0))
    decoded = sum(masses) * unit
    discarded = weight - decoded
    assert 0 <= discarded < len(values) * unit
    tv = (
        sum(
            (abs(F(v) / weight - F(q, sum(masses))) for v, q in zip(values, masses, strict=True)),
            F(0),
        )
        / 2
    )
    assert tv <= discarded / weight


if __name__ == "__main__":
    rows = []
    for exponent in [0, 1, 75, 127, 254]:
        for mantissa in [0, 1, 0x400001, 0x7FFFFF]:
            value = fp32((exponent << 23) | mantissa)
            if value > 0:
                rows.append([value])
    rows += [
        [fp32(1), fp32(2)],
        [1.0, fp32(1)],
        [1.5, 1.0, 2.0**-52],
        [1.0, 2.0**-53],
        [1.0, 3 * 2.0**-53],
    ]
    rng = random.Random(71)
    for n in [1, 2, 3, 7, 64, 1024]:
        for _ in range(10):
            values = [fp32(rng.randrange(0x7F7FFFFF + 1)) for _ in range(n)]
            assert max(values) > 0
            rows.append(values)
    cases = 0
    for values in rows:
        for padding in [0, 1, 8, 29]:
            check(values, padding)
            cases += 1
    masses, unit = encode([1.0, 2.0**-53], 29)
    assert unit == F(1, 2**52) and masses == [2**52, 0]
    # Original-value output divided by encoded total is not encoded q/Q.
    values = [1.0, 3 * 2.0**-53]
    q, unit = encode(values, 29)
    encoded_total = sum(q) * unit
    original_numerator = [F(v) / encoded_total for v in values]
    encoded_law = [F(m, sum(q)) for m in q]
    assert sum(original_numerator) > 1 and original_numerator != encoded_law

    # A normalized, exactly representable FP32 row has a low positive mass lost
    # by the later uint64->double prefix comparison, independently of the codec.
    delta = F(1, 2**54)
    exact_values = [F(1, 4)] * 3 + [
        F(1, 4) - F(1, 2**26),
        F(1, 2**26) - F(1, 2**50),
        F(1, 2**50) - delta,
        delta,
    ]
    values = [float(x) for x in exact_values]
    assert [F(fp32(bits(v))) for v in values] == exact_values
    assert sum(exact_values) == 1
    q, unit = encode(values, 29)
    assert unit == delta and q[-1] == 1 and sum(q) == 2**54
    assert all(F(v) == m * unit for v, m in zip(values, q, strict=True))
    prefix = list(itertools.accumulate(q))
    exclusive = [0, *prefix[:-1]]
    target = float(sum(q))  # p_eff1, exactly as the source double comparison.
    kept = [target > float(e) and m > 0 for e, m in zip(exclusive, q, strict=True)]
    assert not kept[-1] and q[-1] > 0
    assert float(2**54 - 1) == float(2**54)
    report = {
        "kind": "row_relative_fixed_point_mass",
        "status": "passed",
        "native_execution": False,
        "codec_and_tv_cases": cases,
        "padding_widths": [0, 1, 8, 29],
        "largest_row": 1024,
        "drop_witness": {"maximum": 1, "value": "2**-53", "unit": "2**-52", "mass": 0},
        "original_numerator_output": {
            "sum": str(sum(original_numerator)),
            "differs_from_encoded_distribution": True,
        },
        "p1_double_boundary": {
            "normalized_input": [str(x) for x in exact_values],
            "total_integer_mass": sum(q),
            "last_integer_mass": q[-1],
            "last_retained_by_source_comparison": kept[-1],
        },
        "boundary": "Logical codec/error and literal arithmetic only; no native CUDA/RNG run",
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
