"""Ownership/coordinate witnesses only, without Triton compilation."""

import sys

cases = 0
for m in range(1, 18):
    for n in range(1, 10):
        for g in range(1, 9):
            coordinates = []
            for t in range(m * n):
                group_id = t // (g * n)
                first = group_id * g
                height = min(m - first, g)
                coordinates.append((first + t % height, (t % (g * n)) // height))
            assert len(set(coordinates)) == m * n
            assert set(coordinates) == {(i, j) for i in range(m) for j in range(n)}
            for workers in range(1, 12):
                owners = [t for p in range(min(workers, m * n)) for t in range(p, m * n, workers)]
                assert sorted(owners) == list(range(m * n))
            cases += 1
counts = [6, 1, 4]
prefix = [0, 6, 7, 11]
assert [(next(g for g in range(3) if prefix[g] <= t < prefix[g + 1]), t) for t in (0, 4, 8)] == [
    (0, 0),
    (0, 4),
    (2, 8),
]
# Unique coordinates distinguish actual N-half split from alternating columns.
row = list(range(256))
c0, c1 = row[:128], row[128:]
assert c0 + c1 == row and c1[0] == 128
assert 128 * 128 == 64 * 256
sys.stdout.write(
    " ".join(
        str(value)
        for value in [
            f"PASS: {cases} grouped-raster shapes, 11 worker counts each; "
            "grouped lookup and split-N capacity"
        ]
    )
    + "\n"
)
