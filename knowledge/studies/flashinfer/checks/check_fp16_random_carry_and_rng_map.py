"FP16 source software law versus rational grid; logical conversion random addresses."

import bisect
import json
import struct
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

MASK = (1 << 32) - 1
A, C = 1664525, 1013904223
A4, C4 = A**4 & MASK, C * (A**3 + A**2 + A + 1) & MASK


def fp32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def software(bits, random13):
    magnitude, sign = bits & 0x7FFFFFFF, (bits >> 16) & 0x8000
    exponent = magnitude >> 23
    mantissa = (magnitude & 0x7FFFFF) | 0x800000
    if exponent >= 113:
        prepared = magnitude - (112 << 23)
    elif exponent >= 90:
        prepared = mantissa >> (113 - exponent)
    else:
        prepared = 0
    result = min((prepared + (random13 & 8191)) >> 13, 0x7C00)
    if magnitude >= 0x7F800000:
        result = 0x7E00 if magnitude > 0x7F800000 else 0x7C00
    return sign | result


def half(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def philox(seed, offset, rounds):
    c0, c1, c2, c3 = offset & MASK, (offset >> 32) & MASK, 0, 0
    k0, k1 = seed & MASK, (seed >> 32) & MASK
    for _ in range(rounds):
        pa, pb = c0 * 0xD2511F53, c2 * 0xCD9E8D57
        c0, c1, c2, c3 = (
            ((pb >> 32) ^ c1 ^ k0) & MASK,
            pb & MASK,
            ((pa >> 32) ^ c3 ^ k1) & MASK,
            pa & MASK,
        )
        k0, k1 = (k0 + 0x9E3779B9) & MASK, (k1 + 0xBB67AE85) & MASK
    return c0, c1, c2, c3


def peeled_two(seed, offset):
    k0, k1 = seed & MASK, (seed >> 32) & MASK
    p0 = (offset & MASK) * 0xD2511F53
    x0 = ((offset >> 32) & MASK) ^ k0
    pa2 = x0 * 0xD2511F53
    c2_first = (p0 >> 32) ^ k1
    pb = c2_first * 0xCD9E8D57
    return (
        (pb >> 32) ^ ((k0 + 0x9E3779B9) & MASK),
        pb & MASK,
        (pa2 >> 32) ^ ((k1 + 0xBB67AE85) & MASK) ^ (p0 & MASK),
        pa2 & MASK,
    )


if __name__ == "__main__":
    # Exact half grid is independent of the source bit-rebase formula.
    grid = [F(half(bits)) for bits in range(0x7C00)]
    normal_cases = subnormal_cases = 0
    for exponent in [0, 1, 89, 90, 91, 100, 112, 113, 114, 126, 127, 128, 141, 142]:
        for mantissa in [0, 1, 0x1000, 0x1001, 0x3FFFFF, 0x7FFFFF]:
            magnitude = (exponent << 23) | mantissa
            x = F(fp32(magnitude))
            if x > grid[-1]:
                continue
            prepared = F((x * 2**37).__floor__(), 2**37) if x < F(1, 2**14) else x
            lo = bisect.bisect_right(grid, prepared) - 1
            hi = lo + (grid[lo] != prepared)
            probability = (prepared - grid[lo]) / (grid[hi] - grid[lo]) if hi != lo else F(0)
            away_count = probability * 8192
            assert away_count.denominator == 1
            for sign in [0, 0x80000000]:
                counts = Counter(software(magnitude | sign, r) for r in range(8192))
                sign16 = 0x8000 if sign else 0
                expected = Counter({lo | sign16: 8192 - int(away_count)})
                if away_count:
                    expected[hi | sign16] += int(away_count)
                assert counts == expected
                mean = sum((F(half(bits)) * count for bits, count in counts.items()), F(0)) / 8192
                assert mean == (-prepared if sign else prepared)
                if x < F(1, 2**14):
                    assert 0 <= x - prepared < F(1, 2**37)
                    subnormal_cases += 1
                else:
                    assert prepared == x
                    normal_cases += 1
    tiny_bits = 89 << 23  # exactly2^-38; source software emits zero for every random field.
    assert all(software(tiny_bits, r) == 0 for r in range(8192))
    special_cases = 0
    for bits in [0x7F800000, 0xFF800000, 0x7FC00001, 0xFFC00001, 0x7F7FFFFF]:
        for r in [0, 1, 4096, 8191]:
            result = software(bits, r)
            assert result & 0x7FFF <= 0x7E00
            if bits & 0x7FFFFFFF == 0x7F800000:
                assert result & 0x7FFF == 0x7C00
            elif bits & 0x7FFFFFFF > 0x7F800000:
                assert result & 0x7FFF == 0x7E00
            special_cases += 1
    # Flat logical state coordinate stays stable under load regrouping and CTA head splits.
    seed = 0x123456789ABCDEF0
    slots = [0, 3, 65536, 65537]
    address_cases = high_counters = 0
    canonical_pool = {}
    for slot in slots:
        for head in range(4):
            for row in [0, 1, 7, 127]:
                for part in range(2):
                    off_row = ((slot * 4 + head) * 128 + row) * 128 + part * 64
                    canonical = {}
                    for col in range(64):
                        counter = off_row + (col // 8) * 8
                        component, field_shift = (col % 8) // 2, 16 * (col % 2)
                        word = philox(seed, counter, 10)[component]
                        canonical[col] = (
                            counter,
                            component,
                            field_shift,
                            (word >> field_shift) & 8191,
                        )
                    for col, entry in canonical.items():
                        canonical_pool[slot, head, row, part * 64 + col] = entry[:3]
                    for load in [16, 32]:
                        mapped = {}
                        for group in range(64 // load):
                            for stream in range(load // 8):
                                counter = off_row + group * load + stream * 8
                                assert (counter & MASK) >= (off_row & MASK)
                                for pair in range(4):
                                    word = philox(seed, counter, 10)[pair]
                                    for element in range(2):
                                        col = group * load + stream * 8 + pair * 2 + element
                                        mapped[col] = (
                                            counter,
                                            pair,
                                            element * 16,
                                            (word >> (element * 16)) & 8191,
                                        )
                        assert mapped == canonical
                        address_cases += 64
                    high_counters += off_row >> 32 != 0
    cta_traversals = 0
    for hpc in [1, 2, 4]:
        for request_order in [slots, list(reversed(slots))]:
            observed = {}
            for slot in request_order:
                for subgroup in range(4 // hpc):
                    for hh in range(hpc):
                        head = subgroup * hpc + hh
                        for row in [0, 1, 7, 127]:
                            for col in range(128):
                                flat = ((slot * 4 + head) * 128 + row) * 128
                                observed[slot, head, row, col] = (
                                    flat + (col // 8) * 8,
                                    (col % 8) // 2,
                                    (col % 2) * 16,
                                )
            assert observed == canonical_pool
            cta_traversals += 1
    peeled_cases = 0
    for offset in [0, 8, (1 << 32) - 64, (1 << 32) + 8, (1 << 48) + 24]:
        for key in [0, seed, (1 << 64) - 1]:
            assert peeled_two(key, offset) == philox(key, offset, 2)
            peeled_cases += 1
    # Four jump-ahead chains equal serial LCG word coordinates, not independent streams.
    jump_cases = 0
    for start in [0, 1, 123456789, MASK]:
        words, state = [], start
        for _ in range(32):
            words.append(state)
            state = (A * state + C) & MASK
        lanes = words[:4]
        for group in range(8):
            assert lanes == words[group * 4 : group * 4 + 4]
            for state in lanes:
                folded = state ^ (state >> 16)
                assert folded ^ (folded >> 16) == state  # Bijective bit fold.
            lanes = [(A4 * state + C4) & MASK for state in lanes]
            jump_cases += 1
    # A fixed seed/counter repeats; freshness is a caller-owned epoch choice.
    assert philox(seed, 0, 10) == philox(seed, 0, 10)
    result = {
        "passed": True,
        "native_execution": False,
        "normal_uniform_mean_cases": normal_cases,
        "subnormal_prepared_mean_cases": subnormal_cases,
        "special_cases": special_cases,
        "regrouped_element_addresses": address_cases,
        "rows_above_32bit_counter": high_counters,
        "cta_head_request_traversals": cta_traversals,
        "peeled_two_round_cases": peeled_cases,
        "lcg_jumpahead_cases": jump_cases,
        "counterexample": "FP32 value2^-38 always rounds to zero in source FP16 software law",
        "open": [
            "GPU cvt.rs subnormal/special equivalence",
            "Philox/LCG conditional statistical law",
            "native preconversion fold arithmetic",
            "seed epoch advancement across layer/step",
        ],
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
