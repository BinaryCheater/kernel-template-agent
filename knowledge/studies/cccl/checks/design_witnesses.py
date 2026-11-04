import itertools
import sys
from bisect import bisect_left, bisect_right
from collections import Counter

"""Semantic witnesses distinguishing stable ties and histogram/radix mapping."""


A = [(1, "A0"), (1, "A1"), (4, "A2")]
B = [(1, "B0"), (2, "B1"), (4, "B2"), (5, "B3")]
expected = sorted(A + B, key=lambda x: x[0])
for d in range(len(expected) + 1):
    lo = max(0, d - len(B))
    hi = min(d, len(A))
    while lo < hi:
        mid = (lo + hi) // 2
        if B[d - 1 - mid][0] < A[mid][0]:
            hi = mid
        else:
            lo = mid + 1
    assert sorted(A[:lo] + B[: d - lo], key=lambda x: x[0]) == expected[:d]
# Stable digit scatter includes identity payloads and uneven tile lengths.
tiles = [[(1, "a"), (0, "b"), (1, "c")], [(0, "d"), (1, "e")]]
counts = [Counter(x[0] for x in t) for t in tiles]
base = {0: 0, 1: sum(c[0] for c in counts)}
out = [None] * 5
for ti, t in enumerate(tiles):
    local = Counter()
    for digit, payload in t:
        pos = base[digit] + sum(c[digit] for c in counts[:ti]) + local[digit]
        assert out[pos] is None
        out[pos] = (digit, payload)
        local[digit] += 1
assert out == sorted(tiles[0] + tiles[1], key=lambda x: x[0])
bins = [2, 2, -1, 5, 5, 5, 2]
coalesced = Counter()
last = None
run = 0
for b in [*bins, None]:
    if b != last:
        if last is not None and last >= 0:
            coalesced[last] += run
        last = b
        run = 0
    run += 1
assert coalesced == Counter(b for b in bins if b >= 0)
sys.stdout.write(
    " ".join(
        str(value)
        for value in [
            (
                "PASS: stable merge diagonals with duplicate keys; radix key/paylo"
                "ad scatter; decoded-bin run conservation"
            )
        ]
    )
    + "\n"
)

A = [1, 1, 1, 2, 4, 4]
B = [1, 1, 3, 4, 4, 4]


def balanced(d):
    lo = max(0, d - len(B))
    hi = min(d, len(A))
    while lo < hi:
        mid = (lo + hi) // 2
        if B[d - 1 - mid] < A[mid]:
            hi = mid
        else:
            lo = mid + 1
    i = lo
    j = d - i
    star = 0
    if j < len(B):
        x = B[j]
        a0 = bisect_left(A, x, 0, i)
        b0 = bisect_left(B, x, 0, j)
        run1 = i - a0
        total = run1 + j - b0
        adv2 = max(total // 2, total - run1)
        end2 = min(len(B), b0 + adv2 + 1)
        run2 = bisect_right(B, x, j, end2) - b0
        adv2 = min(adv2, run2)
        adv1 = total - adv2
        star = int(adv1 == adv2 + 1 and adv2 < run2)
        i = a0 + adv1
    return i, d - i + star


for stride in (1, 2, 3, 5):
    cuts = [balanced(d) for d in range(0, len(A) + len(B), stride)] + [(len(A), len(B))]
    inter = Counter()
    diff = Counter()
    for (a, b), (aa, bb) in itertools.pairwise(cuts):
        ca, cb = Counter(A[a:aa]), Counter(B[b:bb])
        inter += ca & cb
        diff += ca - cb
    assert inter == Counter(A) & Counter(B)
    assert diff == Counter(A) - Counter(B)
sys.stdout.write(
    " ".join(
        str(value)
        for value in [
            (
                "PASS: duplicate-balanced cuts preserve intersection and differenc"
                "e for four tile spacings"
            )
        ]
    )
    + "\n"
)
mask_lanes = [1, 4, 9]
state = ["A", "B", "C"]
stride = 1
while stride < 4:
    old = state[:]
    state = [v + (old[r + stride] if r + stride < len(old) else "") for r, v in enumerate(old)]
    stride *= 2
assert state[0] == "ABC"
assert [sum(x < lane for x in mask_lanes) for lane in mask_lanes] == [0, 1, 2]
sys.stdout.write(
    " ".join(
        str(value)
        for value in [
            ("PASS: sparse three-lane logical-rank reduction preserves noncommutative ABC order")
        ]
    )
    + "\n"
)
