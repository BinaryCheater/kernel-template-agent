"Exact rational scalar-decay update log; includes accepted-prefix and reset witnesses."

import json
from fractions import Fraction as F
from pathlib import Path

s0 = [[F(2), F(3)], [F(5), F(7)]]
records = [
    (F(1, 2), [F(1), F(3)], [F(2), F(1)]),
    (F(3, 4), [F(-2), F(1)], [F(1), F(-1)]),
    (F(1, 3), [F(4), F(-1)], [F(-1), F(2)]),
]


def step(s, r):
    a, d, k = r
    return [[a * s[v][j] + d[v] * k[j] for j in range(2)] for v in range(2)]


def decode(n):
    out = [[x for x in row] for row in s0]
    for a, _, _ in records[:n]:
        out = [[a * x for x in row] for row in out]
    for j, (_, d, k) in enumerate(records[:n]):
        decay = F(1)
        for a, _, _ in records[j + 1 : n]:
            decay *= a
        out = [[out[v][z] + decay * d[v] * k[z] for z in range(2)] for v in range(2)]
    return out


s = s0
for n in range(4):
    assert s == decode(n)
    q = [F(2), F(-3)]
    assert [sum(row[j] * q[j] for j in range(2)) for row in s] == [
        sum(row[j] * q[j] for j in range(2)) for row in decode(n)
    ]
    if n < 3:
        s = step(s, records[n])
assert decode(1) != decode(3)
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "outcome": "passed",
            "accepted_counts": [0, 1, 2, 3],
            "scope": (
                "Exact rational2x2 update/log identity and prefix distinction; no "
                "native rounding or publication assurance"
            ),
        },
        indent=2,
    )
    + "\n"
)
