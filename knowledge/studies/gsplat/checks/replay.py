"Focused exact-arithmetic checks of source-derived invariants; no GPU execution."

import json
from fractions import Fraction as F
from pathlib import Path


def summary(items):
    color, trans = F(0), F(1)
    for alpha, value in items:
        color += trans * alpha * value
        trans *= 1 - alpha
    return color, trans


def compose(a, b):
    return a[0] + a[1] * b[0], a[1] * b[1]


def main():
    a = [(F(1, 2), F(3)), (F(1, 4), F(7)), (F(3, 4), F(11))]
    assert compose(summary(a[:1]), summary(a[1:])) == summary(a)
    assert compose(summary(a[:1]), summary(a[1:])) != compose(summary(a[1:]), summary(a[:1]))
    final_color, final_t = summary(a)
    trans, suffix = final_t, F(0)
    for i in reversed(range(len(a))):
        alpha, color = a[i]
        trans /= 1 - alpha
        assert trans == summary(a[:i])[1]
        derivative = trans * color - suffix / (1 - alpha)
        changed = a.copy()
        changed[i] = (alpha + F(1, 1000), color)
        assert (summary(changed)[0] - final_color) * 1000 == derivative
        suffix += trans * alpha * color
    # A crossing summary cannot be folded as an exact exclusive-stop prefix.
    threshold = F(1, 10)
    incoming = F(1, 2)
    crossing = [(F(1, 2), F(1)), (F(9, 10), F(8))]
    assert incoming * summary(crossing)[1] <= threshold
    assert incoming * summary(crossing[:1])[1] > threshold
    assert summary(crossing) != summary(crossing[:1])
    weights = [F(1, 5), F(3, 10), F(1, 4)]
    depths = [F(2), F(5), F(9)]
    moment = F(0)
    A = B = F(0)
    for w, z in zip(weights, depths, strict=False):
        moment += 2 * w * (z * A - B)
        A += w
        B += w * z
    pair = sum(
        weights[i] * weights[j] * abs(depths[i] - depths[j]) for i in range(3) for j in range(3)
    )
    assert moment == pair
    positions = [0, 7, 31, 32, 63, 127]
    words = [0] * 4
    for pos in positions:
        words[pos // 32] |= 1 << (pos % 32)
    for rank, pos in enumerate(positions):
        assert (
            sum(x.bit_count() for x in words[: pos // 32])
            + (words[pos // 32] & ((1 << (pos % 32)) - 1)).bit_count()
            == rank
        )
    # D=3, lambda=1: weights preserve identity mean and covariance.
    wm = [F(1, 4)] + [F(1, 8)] * 6
    points = [(F(0),) * 3]
    for axis in range(3):
        for sign in (-1, 1):
            points.append(tuple(F(sign * 2) if j == axis else F(0) for j in range(3)))
    assert sum(wm) == 1
    for j in range(3):
        assert sum(w * x[j] for w, x in zip(wm, points, strict=False)) == 0
        assert sum(w * x[j] * x[j] for w, x in zip(wm, points, strict=False)) == 1
    N, D = 7, 5
    record_major = [r * D + f for r in range(N) for f in range(D)]
    field_major = [f * N + r for r in range(N) for f in range(D)]
    assert sorted(record_major) == sorted(field_major) == list(range(N * D))
    upper = [(i, j) for i in range(3) for j in range(i, 3)]
    assert [sum(3 - a for a in range(i)) + (j - i) for i, j in upper] == list(range(6))
    packed_grad = [F(i + 1) for i in range(6)]
    tangent = [[F(10 * min(i, j) + max(i, j) + 1) for j in range(3)] for i in range(3)]
    full_grad = [[F(0)] * 3 for _ in range(3)]
    for g, (i, j) in zip(packed_grad, upper, strict=False):
        full_grad[i][j] = g if i == j else g / 2
        full_grad[j][i] = full_grad[i][j]
    assert sum(g * tangent[i][j] for g, (i, j) in zip(packed_grad, upper, strict=False)) == sum(
        full_grad[i][j] * tangent[i][j] for i in range(3) for j in range(3)
    )
    result = {
        "outcome": "passed",
        "checks": [
            "ordered summary and noncommutativity",
            "reverse transmittance and exact color derivative",
            "threshold crossing counterexample",
            "sorted pair-distance moments",
            "bit-rank boundaries",
            "sigma identity moments",
            "record/field axis bijections",
            "triangular coordinate and symmetric tangent derivative",
        ],
        "scope": "local algebra only; not CUDA execution, geometric coverage or a ProofResult",
    }
    Path(__file__).with_name("replay-result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
