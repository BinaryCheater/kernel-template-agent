"""Finite semantic witnesses; no CUDA execution or native ordering claim."""

import sys
from itertools import product
from math import prod
from random import Random

rng = Random(20261104)
# Compare bounded opposite-order merges against an independent full-sort oracle.
for capacity in (1, 2, 4, 8, 16):
    for blocks in (1, 2, 3, 9):
        keys = [(rng.randrange(7), i) for i in range(capacity * blocks)]
        retained = sorted(keys[:capacity])
        for start in range(capacity, len(keys), capacity):
            incoming = sorted(keys[start : start + capacity], reverse=True)
            retained = sorted(max(a, b) for a, b in zip(retained, incoming, strict=False))
        assert retained == sorted(keys)[-capacity:]


# Recover every packed bit from a unique incidence map, then test byte patterns.
def pack4(bs):
    out = 0
    for h in range(2):
        q = [(b >> (4 * h)) & 15 for b in bs]

        def c(z):
            return ((z & 8) << 12) | ((z & 7) << 6)

        def d(z):
            return ((z & 8) << 11) | ((z & 6) << 9) | ((z & 1) << 13)

        out |= (c(q[0]) | (c(q[1]) >> 3) | (c(q[2]) >> 6) | d(q[3])) << (16 * h)
    return out


incidence = []
for bit in range(32):
    bs = [0] * 4
    bs[bit // 8] = 1 << (bit % 8)
    value = pack4(bs)
    assert value.bit_count() == 1
    incidence.append(value.bit_length() - 1)
assert sorted(incidence) == list(range(32))
for pos, byte in product(range(4), range(256)):
    bs = [0] * 4
    bs[pos] = byte
    packed = pack4(bs)
    recovered = sum(((packed >> dest) & 1) << src for src, dest in enumerate(incidence))
    assert recovered == byte << (8 * pos)
# Hopper scale pointer formula must equal independently permuted digit address.
for warps in (1, 2, 4, 8):
    m, k = 32 * warps * 2, 6
    dims = [m // (32 * warps), 2, warps, 2, 8, k // 2, 2]
    order = [0, 2, 5, 1, 4, 6, 3]
    seen = set()
    for digits in product(*(range(d) for d in dims)):
        row = (((digits[0] * 2 + digits[1]) * warps + digits[2]) * 2 + digits[3]) * 8 + digits[4]
        col = digits[5] * 2 + digits[6]
        address = 0
        for i in order:
            address = address * dims[i] + digits[i]
        outer = row // (32 * warps) * warps + row // 16 % warps
        inner = (
            col // 2 * 64 + row // (16 * warps) % 2 * 32 + row % 8 * 4 + col % 2 * 2 + row // 8 % 2
        )
        assert address == outer * (k * 32) + inner
        seen.add(address)
    assert seen == set(range(m * k))
# The two Hopper value digit orders have an exact finite inverse and no alias.
for u in (1, 4):
    dims = [2, 1, 2, 4, 2, 1, 3, 4 // u, 2, 4, 1, u]
    order = [0, 3, 6, 10, 4, 9, 7, 1, 8, 2, 5, 11]
    seen = set()
    for digits in product(*(range(d) for d in dims)):
        address = 0
        for i in order:
            address = address * dims[i] + digits[i]
        seen.add(address)
    assert seen == set(range(prod(dims)))


# Packed-column count compared with independent per-bit count, including maxima.
def vpopc(words):
    n = len(words)
    a1, a2 = min(8, n), min(16, n // min(8, n))
    first = [
        [sum((w >> s) & 0x11111111 for w in words[i : i + a1]) for s in range(4)]
        for i in range(0, n, a1)
    ]
    second = [
        [
            [sum((v[s] >> (4 * t)) & 0x0F0F0F0F for v in first[i : i + a2]) for s in range(4)]
            for t in range(2)
        ]
        for i in range(0, len(first), a2)
    ]
    return [
        sum((v[t][s] >> (8 * u)) & 255 for v in second)
        for u in range(4)
        for t in range(2)
        for s in range(4)
    ]


for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
    for words in ([0xFFFFFFFF] * n, [0] * n, [rng.getrandbits(32) for _ in range(n)]):
        assert vpopc(words) == [sum((w >> b) & 1 for w in words) for b in range(32)]
# Sparse row planner: each CTA permutes its own interleaved enumeration only.
for rows, ctas, row_tile, local_tiles in ((37, 3, 4, 4), (8, 2, 2, 2), (1, 1, 4, 1)):
    all_rows = []
    for cta in range(ctas):
        enumeration = [
            ((t * ctas + cta) * row_tile + u) for t in range(local_tiles) for u in range(row_tile)
        ]
        enumeration = [r for r in enumeration if r < rows]
        masks = {r: rng.getrandbits(9) for r in enumeration}
        perm = sorted(enumeration, key=lambda r: -min(4, masks[r].bit_count()))
        assert sorted(perm) == enumeration
        for r in perm:
            bits, members = masks[r], []
            while bits:
                low = bits & -bits
                members.append(low.bit_length() - 1)
                bits ^= low
            assert members == [k for k in range(9) if (masks[r] >> k) & 1]
        all_rows.extend(perm)
    assert sorted(all_rows) == list(range(rows))
# Cyclic tile split: inverse owner map covers tails and empty participants.
for length, tile, parts in product(range(73), (1, 3, 8, 16), (1, 2, 5, 11)):
    owners = [
        (s, j, u, (s + j * parts) * tile + u)
        for s in range(parts)
        for j in range((length + tile * parts - 1) // (tile * parts))
        for u in range(tile)
        if (s + j * parts) * tile + u < length
    ]
    assert sorted(x[3] for x in owners) == list(range(length))
    for s, j, u, k in owners:
        assert (k // tile % parts, k // tile // parts, k % tile) == (s, j, u)
sys.stdout.write(
    " ".join(
        str(value)
        for value in [
            (
                "PASS: full-sort top-k oracle; 32-bit FP4 permutation and 1024 byt"
                "e cases; Hopper digit/pointer identity; packed column counts; loc"
                "al row buckets/bitmap order; cyclic split tails and empty owners"
            )
        ]
    )
    + "\n"
)
# Packing-axis exchange is a 2x2 nibble transpose and is its own inverse.
for a, b in product(range(256), repeat=2):
    lo, hi = (a & 15) | ((b & 15) << 4), (a >> 4) | (b & 240)
    assert ((lo & 15) | ((hi & 15) << 4), (lo >> 4) | (hi & 240)) == (a, b)
sys.stdout.write(
    " ".join(str(value) for value in ["PASS: 65536 FP4 packing-axis byte pairs"]) + "\n"
)
# Code maxima are compared with an independent decoded E2M1 value table.
values = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
for codes in product(range(16), repeat=3):
    assert values[max(q & 7 for q in codes)] == max(
        abs((-1 if q & 8 else 1) * values[q & 7]) for q in codes
    )
sys.stdout.write(
    " ".join(str(value) for value in ["PASS: 4096 signed-code magnitude maxima"]) + "\n"
)
