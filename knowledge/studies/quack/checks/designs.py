import math
import struct
import sys

for K in [64, 128]:
    coords = []
    for t in range(128):
        for j in range(K // 2):
            b, p, e = j >> 3, (j >> 1) & 3, j & 1
            coords.append(
                (
                    16 * (t // 32) + (t % 32) // 4 + 8 * (p % 2),
                    16 * b + 2 * (t % 4) + 8 * (p // 2) + e,
                )
            )
    assert len(set(coords)) == 64 * K and set(coords) == {
        (r, c) for r in range(64) for c in range(K)
    }
# Cyclic windows compare whole-bitstream extraction to adjacent-word extraction.
words = [0x01234567, 0x89ABCDEF, 0xFEDCBA98, 0x76543210]
bits = sum(w << (32 * i) for i, w in enumerate(words))
for j in range(32):
    off = 4 * j
    b, sh = divmod(off, 32)
    observed = ((words[b] | words[(b + 1) % 4] << 32) >> sh) & 65535
    expected = sum(((bits >> ((off + k) % 128)) & 1) << k for k in range(16))
    assert observed == expected
for n in [1, 2, 3, 31, 128, 1024]:
    out = []
    for idx in range(n * (n + 1) // 2):
        r = math.ceil(math.sqrt(2 * idx + 2.25) - 0.5) - 1
        c = idx - r * (r + 1) // 2
        assert 0 <= c <= r < n
        out.append((r, c))
    assert len(set(out)) == len(out)


# Widened fp16 normals retain 13 vacant bits; use 12 as N4096.
def u(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f(x):
    return struct.unpack("<f", struct.pack("<I", x))[0]


vals = [2.0, -2.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5]
mask = 4095
enc = []
for i, x in enumerate(vals):
    assert u(x) & mask == 0
    tag = (~i if x >= 0 else i) & mask
    enc.append((f(u(x) | tag), i))
rank = [i for x, i in sorted(enc, reverse=True)]
assert rank == sorted(range(len(vals)), key=lambda i: (-vals[i], i))
for d in [0, 1, 32768, 65535]:
    assert sum(d + r >= 65536 for r in range(65536)) == d
sys.stdout.write(
    " ".join(
        str(value)
        for value in [
            (
                "PASS: QTIP64/128 coordinate bijection, cyclic seam, triangular bo"
                "undaries, finite nonzero rank tags, BF16 random-carry counts"
            )
        ]
    )
    + "\n"
)

for s in [1, 2, 8, 128, 4096]:
    for origin, L in [(0, 0), (3, 7), (100, 1), (100003, 257)]:
        B = 2**30
        for q in [0, max(L - 1, 0), L, L + 1]:
            addr = ((B - L + q) * s + B * (2**34 - s) + (origin + L) * s) % (2**64)
            assert addr == (origin + q) * s
            assert ((B - L + q) < B) == (q < L)
for n in [1, 2, 31, 32, 33, 63, 65]:
    lengths = [i % 4 for i in range(n)]
    bound = [0]
    for x in lengths:
        bound.append(bound[-1] + x)
    counts = [(x + 1) // 2 * 3 for x in lengths]
    prefix = [0]
    for x in counts:
        prefix.append(prefix[-1] + x)
    for query in range(prefix[-1]):
        expect = next(i for i in range(n) if prefix[i + 1] > query)
        before = 0
        found = None
        for b in range(0, n, 31):
            local = []
            for lane in range(31):
                g = b + lane
                local.append(((bound[g + 1] - bound[g] + 1) // 2) * 3 if g < n else 0)
            for lane, c in enumerate(local):
                if before <= query < before + c:
                    found = b + lane
                    break
                before += c
            if found is not None:
                break
        assert found == expect
for K in range(1, 10):
    for S in range(1, 12):
        ranges = []
        for i in range(S):
            q, r = divmod(K, S)
            start = i * q + min(i, r)
            count = q + (i < r)
            ranges.extend(range(start, start + count))
        assert ranges == list(range(K))
for i, x in enumerate([0, 0.5, 1, 1.5, 2, 3, 4, 6]):
    h = struct.unpack("<H", struct.pack("<e", x * 2**-14))[0]
    assert h == i * 0x200 and h >> 9 == i
sys.stdout.write(
    " ".join(
        str(value)
        for value in [
            (
                "PASS: ragged address/bounds,31-group lookahead/zero groups, balan"
                "ced empty splits, E2M1 carrier-grid points"
            )
        ]
    )
    + "\n"
)
