"Exact unit-lower RHS witness, checking every equation and prior-block coupling."

import json
from fractions import Fraction as F
from pathlib import Path

n, b, nrhs = 64, 16, 3
a = [[F(((i * 7 + j * 3) % 5) - 2, 32) if j < i else F(0) for j in range(n)] for i in range(n)]
rhs = [[F((i + 1) * (c + 2), 7) for c in range(nrhs)] for i in range(n)]
x = [[F(0) for _ in range(nrhs)] for _ in range(n)]
for start in range(0, n, b):
    block = [
        [rhs[i][c] - sum(a[i][j] * x[j][c] for j in range(start)) for c in range(nrhs)]
        for i in range(start, start + b)
    ]
    for local in range(b):
        i = start + local
        for c in range(nrhs):
            x[i][c] = block[local][c] - sum(a[i][j] * x[j][c] for j in range(start, i))
for i in range(n):
    for c in range(nrhs):
        assert x[i][c] + sum(a[i][j] * x[j][c] for j in range(i)) == rhs[i][c]
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "outcome": "passed",
            "order": n,
            "block_order": b,
            "rhs_columns": nrhs,
            "scope": (
                "Exact rational source-shaped algebra; no BF16/TF32 error or nativ"
                "e synchronization check"
            ),
        },
        indent=2,
    )
    + "\n"
)
