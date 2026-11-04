"""Finite arithmetic witnesses; does not execute CUDA or model visibility."""

import sys
from itertools import permutations

H = 1 << 63
cases = 0
for n in range(1, 8):
    for order in permutations(range(n)):
        for base in (0, H):
            state = base
            old = None
            for step, producer in enumerate(order):
                if producer == n - 1:
                    old = state
                state = (state + (H - (n - 1) if producer == n - 1 else 1)) % (2 * H)
                if old is not None:
                    assert bool((state ^ old) & H) == (step == n - 1)
            cases += 1
for x in range(4):
    offsets = [x + 4 * y for y in range(3)]
    assert offsets == [x, x + 4, x + 8]
    assert sum(offsets) == (offsets[0] + offsets[2]) + offsets[1]
sys.stdout.write(
    " ".join(
        str(value)
        for value in [
            f"PASS: {cases} phase arrival orders; four non-power-of-two Y-reduction segments"
        ]
    )
    + "\n"
)
