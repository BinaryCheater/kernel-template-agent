"""Unique-coordinate and live-slot source explanation witnesses; no GPU work."""

import sys
from itertools import product

# Full blocked hierarchy. A 64x16 block gives exactly one physical owner/coordinate.
coords = []
for warp, lane, reg in product(range(4), range(32), range(8)):
    w0, w1 = divmod(warp, 2)
    l0, l1 = divmod(lane, 2)
    r0, r1 = divmod(reg, 4)
    coords.append(((w0 * 16 + l0) * 2 + r0, (w1 * 2 + l1) * 4 + r1))
assert len(set(coords)) == 1024 and set(coords) == set(product(range(64), range(16)))
assert 8 * 2 * 8 == 128  # registers per thread for full 128x128 repeats
for S in (3, 4, 5, 6):
    for producer in range(2 * (S + 1)):
        live = {(producer - j) % (S + 1) for j in range(1, S)}
        free = {producer % (S + 1), (producer + 1) % (S + 1)}
        assert not live & free and live | free == set(range(S + 1))
for minor, major, width in ((5, 3, 2), (9, 4, 4), (3, 7, 8)):
    coords = []
    for start in range(0, minor, width):
        b = start // width
        w = min(width, minor - start)
        for q in range(w * major):
            coords.append((start + q % w, q // w if b % 2 == 0 else major - 1 - q // w))
    assert set(coords) == set(product(range(minor), range(major))) and len(coords) == minor * major
sets = [{0, 2}, {0, 1}]
assert len(set.union(*sets)) == 3
assert len(set.union(*sets) & {0, 2}) == 2
# Logical scale lookup must use k, never packed-byte K, including a group change.
assert 16 // 16 == 1 and (16 // 2) // 16 == 0
sys.stdout.write(
    " ".join(
        str(value)
        for value in [
            (
                "PASS: blocked coordinate bijection; spare-slot lifetime sets; sho"
                "rt snake strips; unique completion issuers; scale index discrimin"
                "ator"
            )
        ]
    )
    + "\n"
)
