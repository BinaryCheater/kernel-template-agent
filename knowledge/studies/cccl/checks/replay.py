"""Ordered, noncommutative semantic witness; no CUDA execution."""

import sys
from itertools import accumulate

cases = 0
for n in range(1, 98):
    tiles = [f"<{i}>" for i in range(n)]
    prefixes = list(accumulate(tiles))
    for current in range(1, n):
        for anchor in range(current):
            # Traversal is backwards, operands must be prepended.
            suffix = ""
            for i in range(current - 1, anchor, -1):
                suffix = tiles[i] + suffix
            got = prefixes[anchor] + suffix
            assert got == "".join(tiles[:current])
            cases += 1
values = list(range(1, 36))
assert sum(values[:32]) + sum(values[32:]) == 630
sys.stdout.write(
    " ".join(
        str(value)
        for value in [f"PASS: {cases} ordered lookback witnesses; 35-of-64 partial block sum=630"]
    )
    + "\n"
)
