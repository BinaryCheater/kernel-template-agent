"Local source-derived LogFMT word layout, strided chunk and copy-stream lag checks."

import json
import math
import random
from pathlib import Path

rng = random.Random(19)
for _ in range(100):
    q = [rng.randrange(512) for _ in range(16)]
    signs = rng.randrange(1 << 16)
    wordmask = (1 << 32) - 1
    words = [
        q[0] | q[1] << 9 | q[2] << 18 | q[3] << 27,
        q[3] >> 5 | q[4] << 4 | q[5] << 13 | q[6] << 22 | q[7] << 31,
        q[7] >> 1 | q[8] << 8 | q[9] << 17 | q[10] << 26,
        q[10] >> 6 | q[11] << 3 | q[12] << 12 | q[13] << 21 | q[14] << 30,
        q[14] >> 2 | q[15] << 7 | signs << 16,
    ]
    packed = sum((w & wordmask) << (32 * i) for i, w in enumerate(words))
    assert [packed >> (9 * i) & 511 for i in range(16)] == q
    assert packed >> 144 == signs
# Codebook center uses q-1; maximum code511 reaches amax, code0 is reservedzero.
Lmin, Lmax = -20.0, -1.0
step = (Lmax - Lmin) / 510
rounding = 2 - math.log2((1 + 2**step) / 2) / step
for q in [1, 2, 17, 255, 511]:
    x = 2 ** ((q - 1) * step + Lmin)
    encoded = math.floor(max(math.log2(x) / step + rounding - Lmin / step, 0))
    assert encoded == q
# Piece splitting of a noncontiguous2x3 outer region, contiguous inner5bytes.
counts = [2, 3]
srcstrides = [100, 11]
dststrides = [200, 17]
inner = 5


def offset(i, strides):
    coords = [i // counts[1], i % counts[1]]
    return sum(c * s for c, s in zip(coords, strides, strict=False))


for b, n in [(0, 30), (3, 19), (4, 2), (5, 5), (23, 7)]:
    remain = n
    it = b // inner
    within = b % inner
    got = []
    while remain:
        take = min(inner - within, remain)
        got.extend(
            (offset(it, srcstrides) + within + j, offset(it, dststrides) + within + j)
            for j in range(take)
        )
        remain -= take
        it += 1
        within = 0
    expected = [
        (offset(k // inner, srcstrides) + k % inner, offset(k // inner, dststrides) + k % inner)
        for k in range(b, b + n)
    ]
    assert got == expected
# At lag=R, first drain is behind producerR reuse wait: symmetric peers form a wait cycle.
for R in range(1, 9):
    for lag in range(R + 1):
        schedule = []
        for n in range(2 * R + 2):
            schedule.append(("producer", n))
            if n >= lag:
                schedule.append(("drain", n - lag))
        assert (schedule.index(("drain", 0)) < schedule.index(("producer", R))) == (lag < R)
Path(__file__).with_name("layout-result.json").write_text(
    json.dumps(
        {
            "outcome": "passed",
            "checks": [
                "100 identifiable16-magnitude/sign pack inversions",
                "log code centers and endpoint",
                "five strided chunks including partial inner endpoints",
                "lag<R queue-order witness and lag=R cycle counterexample",
            ],
            "scope": (
                "local arithmetic/layout only; approximate PTX log/exp, provider v"
                "isibility and GPU scheduling are not executed"
            ),
        },
        indent=2,
    )
    + "\n"
)
