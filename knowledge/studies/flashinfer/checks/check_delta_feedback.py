"Exact scalar delta recurrence versus chunk feedback and both contraction orders."

import json
from fractions import Fraction as F
from pathlib import Path


def mm(a, b):
    return [
        [sum((x * y for x, y in zip(row, col, strict=True)), F(0)) for col in zip(*b, strict=True)]
        for row in a
    ]


def add(a, b, sign=1):
    return [[x + sign * y for x, y in zip(r, s, strict=True)] for r, s in zip(a, b, strict=True)]


def solve(lower, rhs):
    # Solve (I+L)X=RHS row by row; never divide by beta.
    x = []
    for i, row in enumerate(rhs):
        x.append(
            [v - sum((lower[i][j] * x[j][c] for j in range(i)), F(0)) for c, v in enumerate(row)]
        )
    return x


def sequential(h, q, k, v, alpha, beta):
    h = [row[:] for row in h]
    outputs = []
    for i in range(len(q)):
        for row in h:
            row[:] = [alpha[i] * x for x in row]
        pred = mm([k[i]], h)[0]
        r = [beta[i] * (x - y) for x, y in zip(v[i], pred, strict=True)]
        h = add(h, [[key * x for x in r] for key in k[i]])
        outputs.append(mm([q[i]], h)[0])
    return outputs, h


def chunk(h, q, k, v, alpha, beta):
    n = len(q)
    e = []
    prefix = F(1)
    for a in alpha:
        prefix *= a
        e.append(prefix)
    d = [[e[i] / e[j] if i >= j else F(0) for j in range(n)] for i in range(n)]

    def dot(x, y):
        return sum((a * b for a, b in zip(x, y, strict=True)), F(0))

    lower = [
        [beta[i] * d[i][j] * dot(k[i], k[j]) if j < i else F(0) for j in range(n)] for i in range(n)
    ]
    rhs = [[beta[i] if i == j else F(0) for j in range(n)] for i in range(n)]
    t = solve(lower, rhs)
    ek = [[e[i] * x for x in k[i]] for i in range(n)]
    eq = [[e[i] * x for x in q[i]] for i in range(n)]
    c = [[d[i][j] * dot(q[i], k[j]) for j in range(n)] for i in range(n)]
    r = mm(t, add(v, mm(ek, h), -1))
    residual_output = add(mm(eq, h), mm(c, r))
    p = mm(c, t)
    a = add(eq, mm(p, ek), -1)
    coefficient_output = add(mm(a, h), mm(p, v))
    carry = [[e[-1] * x for x in row] for row in h]
    outgoing = add(carry, mm([[d[-1][i] * k[i][j] for i in range(n)] for j in range(len(k[0]))], r))
    return residual_output, coefficient_output, outgoing, lower, rhs, t


def two_block_solve(lower, rhs, width):
    # Source's two diagonal solves, lower product and dependent solve.
    t00 = solve([row[:width] for row in lower[:width]], [row[:width] for row in rhs[:width]])
    l11 = [row[width:] for row in lower[width:]]
    t11 = solve(l11, [row[width:] for row in rhs[width:]])
    y = mm([[-x for x in row[:width]] for row in lower[width:]], t00)
    t10 = solve(l11, y)
    zero = [F(0)] * width
    return [r + zero for r in t00] + [a + b for a, b in zip(t10, t11, strict=True)]


def channel_sequential(h, q, k, v, alpha, beta):
    h = [row[:] for row in h]
    outputs = []
    for i in range(len(q)):
        h = [[alpha[i][c] * x for x in row] for c, row in enumerate(h)]
        pred = mm([k[i]], h)[0]
        r = [beta[i] * (x - y) for x, y in zip(v[i], pred, strict=True)]
        h = add(h, [[key * x for x in r] for key in k[i]])
        outputs.append(mm([q[i]], h)[0])
    return outputs, h


def channel_chunk(h, q, k, v, alpha, beta):
    n, width = len(q), len(k[0])
    e = []
    prefix = [F(1)] * width
    for gate in alpha:
        prefix = [x * y for x, y in zip(prefix, gate, strict=True)]
        e.append(prefix)
    ek = [[x * y for x, y in zip(k[i], e[i], strict=True)] for i in range(n)]
    eq = [[x * y for x, y in zip(q[i], e[i], strict=True)] for i in range(n)]
    ki = [[x / y for x, y in zip(k[i], e[i], strict=True)] for i in range(n)]
    pair_k = mm(ek, list(map(list, zip(*ki, strict=True))))
    pair_q = mm(eq, list(map(list, zip(*ki, strict=True))))
    lower = [[beta[i] * pair_k[i][j] if j < i else F(0) for j in range(n)] for i in range(n)]
    causal = [[pair_q[i][j] if j <= i else F(0) for j in range(n)] for i in range(n)]
    t = solve(lower, [[beta[i] if i == j else F(0) for j in range(n)] for i in range(n)])
    r = mm(t, add(v, mm(ek, h), -1))
    y0 = add(mm(eq, h), mm(causal, r))
    p = mm(causal, t)
    y1 = add(mm(add(eq, mm(p, ek), -1), h), mm(p, v))
    out = add(
        [[e[-1][c] * x for x in row] for c, row in enumerate(h)],
        mm([[e[-1][c] * ki[i][c] for i in range(n)] for c in range(width)], r),
    )
    return y0, y1, out


if __name__ == "__main__":
    cases = padded = block_cases = 0
    for n in [1, 2, 3, 4, 5, 7, 8, 16]:
        for recipe in range(6):
            h = [[F(2), F(-1)], [F(1, 3), F(4)]]
            q = [[F((i + recipe) % 5 - 2, 3), F(1, 2)] for i in range(n)]
            k = [[F((i + 1) % 3 - 1, 2), F((recipe + i) % 3, 3)] for i in range(n)]
            v = [[F(i + 1, 3), F(recipe - i, 2)] for i in range(n)]
            alpha = [F(1, 2) if (i + recipe) % 2 else F(1) for i in range(n)]
            beta = [F(0) if (i + recipe) % 4 == 0 else F(1 + recipe % 2, 3) for i in range(n)]
            expected, state = sequential(h, q, k, v, alpha, beta)
            y0, y1, out, lower, rhs, t = chunk(h, q, k, v, alpha, beta)
            assert y0 == expected == y1 and out == state
            cases += 1
            # Two chunks consume the previous inclusive outgoing state.
            expected2, state2 = sequential(state, q, k, v, alpha, beta)
            z0, z1, out2, *_ = chunk(out, q, k, v, alpha, beta)
            assert z0 == expected2 == z1 and out2 == state2
            cases += 1
            # Check native 16 working rows with logical extents4/8/16 and tails.
            tail = 16 - n
            zp = [[F(0), F(0)]] * tail
            p0, p1, po, pl, prhs, pt = chunk(
                h, q + zp, k + zp, v + zp, alpha + [F(1)] * tail, beta + [F(0)] * tail
            )
            assert p0[:n] == expected == p1[:n] and po == state
            assert all(all(x == 0 for x in row) for row in p0[n:])
            padded += 1
            assert two_block_solve(pl, prhs, 8) == pt
            block_cases += 1
    # Missing feedback is consequential even with no decay and a scalar state.
    y, state = sequential(
        [[F(1)]], [[F(1)], [F(1)]], [[F(1)], [F(1)]], [[F(2)], [F(3)]], [F(1), F(1)], [F(1), F(1)]
    )
    assert y == [[F(2)], [F(3)]] and state == [[F(3)]]
    assert state != [[F(6)]]  # Treating beta*v as independent inserts gives6.
    # Beta belongs on each equation's row and on the solve's diagonal RHS.
    lower = [[F(0), F(0)], [F(1, 3), F(0)]]
    rhs = [[F(0), F(0)], [F(0), F(2, 3)]]
    assert solve(lower, rhs) == rhs  # Zero beta is valid; no inverse-beta recovery.
    channel_cases = scalar_reductions = 0
    for n in [1, 3, 8, 16]:
        for recipe in range(4):
            h = [[F(2), F(-1)], [F(3), F(1)]]
            q = [[F(1), F(i + 1, 3)] for i in range(n)]
            k = [[F((i + recipe) % 3, 3), F(1, 2)] for i in range(n)]
            v = [[F(recipe - i), F(i, 3)] for i in range(n)]
            beta = [F((i + recipe) % 3, 3) for i in range(n)]
            gates = [[F(1, 2), F(1, 4)] if i % 2 else [F(1), F(1, 2)] for i in range(n)]
            expected, state = channel_sequential(h, q, k, v, gates, beta)
            y0, y1, out = channel_chunk(h, q, k, v, gates, beta)
            assert y0 == expected == y1 and out == state
            channel_cases += 1
            scalar = [F(1, 2) if i % 2 else F(1) for i in range(n)]
            equal = [[x, x] for x in scalar]
            a0, a1, ast, *_ = chunk(h, q, k, v, scalar, beta)
            b0, b1, bst = channel_chunk(h, q, k, v, equal, beta)
            assert a0 == b0 == a1 == b1 and ast == bst
            scalar_reductions += 1
    # A scalar collapse loses the second channel's independent decay.
    diagonal_y, _ = channel_sequential(
        [[F(1)], [F(1)]], [[F(1), F(1)]], [[F(0), F(0)]], [[F(0)]], [[F(1, 2), F(1, 4)]], [F(0)]
    )
    scalar_y, _ = sequential(
        [[F(1)], [F(1)]], [[F(1), F(1)]], [[F(0), F(0)]], [[F(0)]], [F(1, 2)], [F(0)]
    )
    assert diagonal_y == [[F(3, 4)]] and scalar_y == [[F(1)]]
    result = {
        "passed": True,
        "native_execution": False,
        "key_channel_cases": channel_cases,
        "equal_channel_reductions": scalar_reductions,
        "recurrence_and_two_chunk_cases": cases,
        "neutral_padding_cases": padded,
        "two_8row_weighted_solve_cases": block_cases,
        "oracle": "Token-serial decay/predict/correct with exact rationals",
        "premises": [
            "scalar positive decay",
            "after-update readout",
            "zero beta",
            "inclusive state carry",
            "neutral padded rows",
            "two-block dependency order",
        ],
        "open": [
            "native stage rounding/residuals",
            "scratch retirement and GPU publication",
            "multiple updates per real token",
        ],
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
