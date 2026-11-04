"Independent serial recurrence versus folded chunk factors; labeled workspace maps."

import json
from fractions import Fraction as F
from pathlib import Path

from check_delta_feedback import add, channel_sequential, mm, solve


def transpose(x):
    return list(map(list, zip(*x, strict=True)))


def prepare(q, k, alpha, beta):
    n, width = len(q), len(k[0])
    e = []
    acc = [F(1)] * width
    for gate in alpha:
        acc = [x * y for x, y in zip(acc, gate, strict=True)]
        e.append(acc)
    ek = [[x * y for x, y in zip(k[i], e[i], strict=True)] for i in range(n)]
    eq = [[x * y for x, y in zip(q[i], e[i], strict=True)] for i in range(n)]
    invk = [[x / y for x, y in zip(k[i], e[i], strict=True)] for i in range(n)]
    kk, qk = mm(ek, transpose(invk)), mm(eq, transpose(invk))
    lower = [[beta[i] * kk[i][j] if j < i else F(0) for j in range(n)] for i in range(n)]
    causal = [[qk[i][j] if j <= i else F(0) for j in range(n)] for i in range(n)]
    weighted = solve(lower, [[beta[i] if i == j else F(0) for j in range(n)] for i in range(n)])
    restore = [[e[-1][c] * invk[i][c] for c in range(width)] for i in range(n)]
    # No values or incoming state are accepted by this preparation function.
    return ek, eq, mm(transpose(restore), weighted), mm(causal, weighted), e[-1]


def consume(h, v, factors):
    ek, eq, w, p, diag = factors
    residual_input = add(v, mm(ek, h), -1)
    output = add(mm(eq, h), mm(p, residual_input))
    state = add([[diag[c] * x for x in row] for c, row in enumerate(h)], mm(w, residual_input))
    return output, state


def pairwise_address(row, col):
    byte = (row * 16 + (col ^ 8)) * 2
    return (byte ^ (((byte >> 7) & 1) << 4)) // 2


if __name__ == "__main__":
    cases = 0
    for n in [1, 3, 8, 16]:
        q = [[F(1), F(i + 1, 3)] for i in range(n)]
        k = [[F(i % 3, 3), F(1, 2)] for i in range(n)]
        alpha = [[F(1, 2), F(1, 4)] if i % 2 else [F(1), F(1, 2)] for i in range(n)]
        beta = [F(i % 3, 3) for i in range(n)]
        factors = prepare(q, k, alpha, beta)
        for generation in range(4):
            h = [[F(generation), F(-1)], [F(3), F(generation, 2)]]
            v = [[F(generation - i), F(i, 3)] for i in range(n)]
            expected, expected_state = channel_sequential(h, q, k, v, alpha, beta)
            actual, state = consume(h, v, factors)
            assert actual == expected and state == expected_state
            # Reuse the factors with the new incoming state on a second ordered chunk.
            expected2, expected_state2 = channel_sequential(state, q, k, v, alpha, beta)
            actual2, state2 = consume(state, v, factors)
            assert actual2 == expected2 and state2 == expected_state2
            cases += 2
    # Identifiable full native coordinates, not repeated low-bit values.
    kd = {(t, key ^ 8): (t, key) for t in range(16) for key in range(128)}
    wt = {(t ^ 8, key): (t, key) for t in range(16) for key in range(128)}
    pair = {pairwise_address(i, j): (i, j) for i in range(16) for j in range(16)}
    assert len(kd) == len(wt) == 2048 and len(pair) == 256
    assert set(pair) == set(range(256))
    obsolete_mismatches = 0
    for t in range(16):
        for key in range(128):
            assert kd[t, key ^ 8] == (t, key)
            assert wt[t ^ 8, key] == (t, key)
            # Stale extra XOR can be self-consistent for one reader, but not tcgen05.
            obsolete_address = key ^ 8 ^ ((t & 2) * 16)
            obsolete_mismatches += kd[t, obsolete_address] != (t, key)
    assert obsolete_mismatches == 1024
    for i in range(16):
        for j in range(16):
            assert pair[pairwise_address(i, j)] == (i, j)
    payload = (3 * 16 * 128 + 16 * 16) * 2 + 128 * 4
    assert payload == 13312
    result = {
        "passed": True,
        "native_execution": False,
        "prepared_one_two_chunk_cases": cases,
        "decay_coordinates": len(kd),
        "w_coordinates": len(wt),
        "pairwise_coordinates": len(pair),
        "obsolete_map_mismatches": obsolete_mismatches,
        "logical_factor_bytes_per_chunk_head": payload,
        "oracle": "Token-serial channelwise recurrence with exact rational arithmetic",
        "open": [
            "native factor folding/casts",
            "TMA descriptor and fragment lowering",
            "asynchronous publication and workspace retirement",
        ],
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
