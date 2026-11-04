"""Exact rank witnesses, tails and ties; no GPU execution."""

import json
from itertools import product
from pathlib import Path

cases = 0
for n in range(1, 9):
    for values in product(range(3), repeat=n):
        for width in (1, 2, 3):
            tiles = [values[i : i + width] for i in range(0, n, width)]
            for k in range(1, n + 1):
                chosen = sorted(range(len(tiles)), key=lambda i: max(tiles[i]), reverse=True)[
                    : min(k, len(tiles))
                ]
                union = [v for i in chosen for v in tiles[i]]
                assert sorted(union, reverse=True)[:k] == sorted(values, reverse=True)[:k]
                cases += 1
# Arbitrary upper bounds do not supply witnesses: a low tile can hide a high tile.
assert sorted([0], reverse=True)[:1] != sorted([0, 9], reverse=True)[:1]
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "passed": True,
            "cases": cases,
            "scope": (
                "Exact integer rank cover, tied values and partial tiles; no nativ"
                "e/GPU or timing claim"
            ),
            "loose_bound_counterexample": {
                "tiles": [[0], [9]],
                "upper_bounds": [100, 9],
                "chosen_tile": 0,
            },
        },
        indent=2,
    )
    + "\n"
)
