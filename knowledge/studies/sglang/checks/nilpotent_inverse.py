"Exact integer16x16 witness: all subdiagonals nonzero; each iteration squares error."

import json
from pathlib import Path

n = 16


def mm(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


i = [[int(r == c) for c in range(n)] for r in range(n)]
a = [[((r + c) % 3) + 1 if r > c else 0 for c in range(n)] for r in range(n)]
s = [[i[r][c] + a[r][c] for c in range(n)] for r in range(n)]
x = [[i[r][c] - a[r][c] for c in range(n)] for r in range(n)]
nonzero = []
for step in range(4):
    err = [[i[r][c] - v for c, v in enumerate(row)] for r, row in enumerate(mm(s, x))]
    nonzero.append(sum(v != 0 for row in err for v in row))
    if step < 3:
        prod = mm(mm(x, s), x)
        x = [[2 * x[r][c] - prod[r][c] for c in range(n)] for r in range(n)]
assert nonzero[-1] == 0 and all(v > 0 for v in nonzero[:-1])
assert mm(x, s) == i
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "outcome": "passed",
            "order": 16,
            "nonzero_residual_entries": nonzero,
            "scope": "Exact integer polynomial; no BF16 residual or native MMA mapping assurance",
        },
        indent=2,
    )
    + "\n"
)
