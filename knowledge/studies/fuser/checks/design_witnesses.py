"""Source explanation checks for digit layout and vector reduce-scatter."""

import sys
from itertools import product

for ro, ri, c in [(32, 4, 4), (2, 3, 4)]:
    rows, cols = ro * ri * 2, c * 3
    addresses = []
    for r, k in product(range(rows), range(cols)):
        rt, rout, rint, ct, ci = r // (ro * ri), r % ro, (r // ro) % ri, k // c, k % c
        off = rt * cols * ro * ri + ct * ro * ri * c + rout * ri * c + rint * c + ci
        addresses.append(off)
        assert rt * ro * ri + rint * ro + rout == r and ct * c + ci == k
    assert sorted(addresses) == list(range(rows * cols))
for lanes in (2, 4, 8):
    values = lanes * 2
    state = [[{(lane, v)} for v in range(values)] for lane in range(lanes)]
    bit = lanes // 2
    while bit:
        old = state
        half = len(old[0]) // 2
        state = []
        for lane in range(lanes):
            start = half if lane & bit else 0
            # Peer sends the chunk retained by this lane.
            state.append([old[lane][start + i] | old[lane ^ bit][start + i] for i in range(half)])
        bit //= 2
    for lane in range(lanes):
        assert state[lane] == [
            {(lane_or_matrix, lane * 2 + i) for lane_or_matrix in range(lanes)} for i in range(2)
        ]
sys.stdout.write(
    " ".join(
        str(value)
        for value in [
            ("PASS: two padded digit-layout bijections and 2/4/8-lane reduce-scatter membership")
        ]
    )
    + "\n"
)
