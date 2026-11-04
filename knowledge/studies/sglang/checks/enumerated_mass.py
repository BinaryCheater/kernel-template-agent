"""Exact ideal single-row mass partition, not a native finite-grid sampler."""

import json
from fractions import Fraction as F
from itertools import permutations, product
from pathlib import Path

cases = 0
for n in range(1, 5):
    for weights in product(range(3), repeat=n):
        total = sum(weights)
        if not total:
            continue
        P = [F(w, total) for w in weights]
        for size in range(n + 1):
            for candidates in permutations(range(n), size):
                mass = sum(P[i] for i in candidates)
                R = 1 - mass
                out = [F(0)] * n
                for i in candidates:
                    out[i] += P[i]
                if R:
                    for i in range(n):
                        if i not in candidates:
                            out[i] += R * (P[i] / R)
                assert out == P
                cases += 1
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "passed": True,
            "cases": cases,
            "scope": (
                "Exact normalized rational rows and ordered distinct candidate sub"
                "sets; no GPU or finite-RNG law claim"
            ),
        },
        indent=2,
    )
    + "\n"
)
