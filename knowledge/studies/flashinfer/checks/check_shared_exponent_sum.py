"Shared-exponent integer sum versus an independent exact-rational FP32 floor oracle."

import itertools
import json
import struct
from fractions import Fraction as F
from pathlib import Path


def value(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def source_reconstruct(count, exponent_field, fraction_sum):
    # Transcription of calcFloatValue; caller is responsible for empty-bin bypass.
    extra = fraction_sum >> 23
    if exponent_field:
        extra += count
    n = extra.bit_length()
    exponent_bits = (exponent_field + n + (0 if exponent_field == 0 else -1)) << 23
    if extra:
        move = 23 - (n - 1)
        extra &= (1 << (n - 1)) - 1
        if move > 0:
            mantissa = ((fraction_sum & 0x7FFFFF) >> (n - 1)) | (extra << move)
        else:
            mantissa = extra >> (-move)
    else:
        mantissa = fraction_sum
    return exponent_bits | mantissa


def floor_oracle(exact):
    # Search the ordered positive FP32 number line, independently of the bit-sum codec.
    lo, hi = 0, 0x7F7FFFFF
    assert F(0) <= exact <= F(value(hi))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if F(value(mid)) <= exact:
            lo = mid
        else:
            hi = mid - 1
    return lo


if __name__ == "__main__":
    cases = 0
    for exponent in [0, 1, 100, 126, 127, 200]:
        for n in range(1, 5):
            for mantissas in itertools.combinations_with_replacement(
                [0, 1, 3, 0x400001, 0x7FFFFF], n
            ):
                vals = [value((exponent << 23) | m) for m in mantissas]
                exact = sum(map(F, vals), F(0))
                actual = source_reconstruct(n, exponent, sum(mantissas))
                assert actual == floor_oracle(exact)
                assert F(value(actual)) <= exact
                if actual < 0x7F7FFFFF:
                    assert exact < F(value(actual + 1))
                for order in [mantissas, tuple(reversed(mantissas))]:
                    # A two-CTA partition can merge its independent integer channels.
                    split = len(order) // 2
                    m0, m1 = sum(order[:split]), sum(order[split:])
                    assert source_reconstruct(n, exponent, m0 + m1) == actual
                cases += 1
    assert cases == 750
    high_count_cases = 0
    for exponent in [0, 1, 100, 127, 200]:
        for count in [2**20, 2**30, 2**31 - 1]:
            for mantissa in [0, 1, 0x7FFFFF]:
                exact = count * F(value((exponent << 23) | mantissa))
                assert count * mantissa < 2**64
                assert source_reconstruct(count, exponent, count * mantissa) == floor_oracle(exact)
                high_count_cases += 1
    assert source_reconstruct(1, 254, 0x7FFFFF) == 0x7F7FFFFF
    assert F(value(254 << 23)) * 2 > F(value(0x7F7FFFFF))  # Overflow excluded.
    assert source_reconstruct(0, 127, 0) != 0  # Source wrapper must bypass empty bins.
    assert source_reconstruct(2, 0, 0x7FFFFF + 1) == 1 << 23  # Subnormal carry.
    assert value(source_reconstruct(3, 127, 0)) == 3.0  # Hidden ones, even with M0.
    assert value(source_reconstruct(2, 127, 0)) == 2.0 != 3.0  # Mixed1/2 needs two exponents.

    vals = [value(127 << 23), value((127 << 23) | 3)]
    truncated_bits = source_reconstruct(2, 127, 3)
    nearest_bits = bits(sum(vals))
    assert truncated_bits != nearest_bits
    order_witness = None
    for mantissas in itertools.combinations_with_replacement([0, 1, 3, 0x200001, 0x7FFFFF], 4):
        results = set()
        for order in itertools.permutations(mantissas):
            acc = 0.0
            for m in order:
                acc = value(bits(acc + value((127 << 23) | m)))
            results.add(acc)
        if len(results) > 1:
            order_witness = {
                "fractions": list(mantissas),
                "fp32_order_results": sorted(results),
                "integer_result": value(source_reconstruct(4, 127, sum(mantissas))),
            }
            break
    assert order_witness is not None
    report = {
        "kind": "shared_exponent_integer_sum_oracle",
        "status": "passed",
        "native_execution": False,
        "rational_floor_cases": cases,
        "high_count_cases": high_count_cases,
        "toward_zero_vs_nearest": {
            "toward_zero": value(truncated_bits),
            "nearest_even": value(nearest_bits),
        },
        "floating_order_witness": order_witness,
        "boundaries": [
            "empty bypass",
            "subnormal-to-normal carry",
            "hidden-one counts",
            "shared exponent required",
            "finite range and Int32 count limit",
        ],
        "native_boundary": "Native atomics/publication and downstream crossing remain separate",
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
