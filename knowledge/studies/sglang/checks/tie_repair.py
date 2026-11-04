"""Repair all arbitrary boundary subsets to the same canonical ordering."""

import json
from itertools import combinations, product
from pathlib import Path

cases = 0
for n in range(1, 8):
    for values in product(range(3), repeat=n):
        order = sorted(range(n), key=lambda i: (-values[i], i))
        for k in range(1, n + 1):
            b = values[order[k - 1]]
            strict = [i for i in range(n) if values[i] > b]
            equal = [i for i in range(n) if values[i] == b]
            quota = k - len(strict)
            for subset in combinations(equal, quota):
                upstream = strict + list(subset)
                fixed = [i for i in upstream if values[i] > b] + equal[:quota]
                repaired = sorted(fixed, key=lambda i: (-values[i], i))
                assert repaired == order[:k]
                cases += 1
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "passed": True,
            "cases": cases,
            "scope": (
                "Exact integer values, all allowed boundary subsets through length"
                "7; no native execution"
            ),
        },
        indent=2,
    )
    + "\n"
)
