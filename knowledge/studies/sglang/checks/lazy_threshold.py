"""Exact rational residual upper bound and strict-comparison guard."""

import json
from fractions import Fraction as F
from pathlib import Path

cases = 0
for d in range(2, 17):
    for a in range(d + 1):
        P = [F(a, d), 1 - F(a, d)]
        for b in range(d + 1):
            Q = [F(b, d), 1 - F(b, d)]
            for cnum in range(1, d):
                c = F(cnum, d)
                m = sum(max(c * p - q, 0) for p, q in zip(P, Q, strict=False))
                h = m / (m + 1 - c)
                assert 0 <= h <= c
                for x in (F(0), h, c, F(1), (h + c) / 2):
                    assert (x < h) == (x < c and x < h)
                    cases += 1
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "passed": True,
            "cases": cases,
            "scope": (
                "Exact rational normalized two-token rows and strict guards; no FP"
                "32/GPU distribution claim"
            ),
        },
        indent=2,
    )
    + "\n"
)
