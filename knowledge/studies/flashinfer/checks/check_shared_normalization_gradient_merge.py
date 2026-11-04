"Exact shared Jacobian identities and finite-format normalization stage distinctions."

import json
import struct
from fractions import Fraction as F
from pathlib import Path


def fp32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def bf16(x):
    bits = struct.unpack("<I", struct.pack("<f", fp32(x)))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def halving(values):
    while len(values) > 1:
        half = len(values) // 2
        values = [fp32(values[i] + values[i + half]) for i in range(half)]
    return values[0]


def source_dot(a, b):
    products = [fp32(x * y) for x, y in zip(a, b, strict=True)]
    quadrants = [halving(products[i : i + 32]) for i in range(0, 128, 32)]
    return fp32(fp32(fp32(quadrants[0] + quadrants[1]) + quadrants[2]) + quadrants[3])


def ordered_sum(parts):
    acc = list(parts[0])
    for part in parts[1:]:
        acc = [fp32(a + b) for a, b in zip(acc, part, strict=True)]
    return acc


def pullback(x, r, incoming, rounded_y=False):
    y = [fp32(v * r) for v in x]
    if rounded_y:
        y = [bf16(v) for v in y]
    dot = source_dot(incoming, y)
    return [bf16(fp32(r * fp32(d - fp32(v * dot)))) for d, v in zip(incoming, y, strict=True)]


if __name__ == "__main__":
    exact_cases = fp32_stage_cases = 0
    for width in [2, 4, 8]:
        for group in [1, 2, 3, 5]:
            for recipe in range(4):
                x = [F(i - recipe, 7) for i in range(width)]
                r = F(recipe + 1, 9)
                parts = [[F((i + j) % 5 - 2, 3) for i in range(width)] for j in range(group)]
                total = [sum(p[i] for p in parts) for i in range(width)]
                y = [v * r for v in x]
                s = sum(d * v for d, v in zip(total, y, strict=True))
                actual = [r * (d - v * s) for d, v in zip(total, y, strict=True)]
                # Independent dense Jacobian multiplication, not the projection helper.
                expected = [
                    sum(
                        ((r if i == j else 0) - r**3 * x[i] * x[j]) * total[j] for j in range(width)
                    )
                    for i in range(width)
                ]
                assert actual == expected
                exact_cases += 1
    raw_rounding_witness = None
    for group in [1, 2, 3, 5]:
        for recipe in range(8):
            x = [bf16(((i + recipe) % 13 - 6) / 8) for i in range(128)]
            r = fp32(0.1 + recipe / 19)
            parts = [
                [fp32(((i * (j + 1) + recipe) % 17 - 8) / 11) for i in range(128)]
                for j in range(group)
            ]
            # Independent materialized indexing preserves the same consumer order.
            workspace = {(j, i): parts[j][i] for j in range(group) for i in range(128)}
            reduced = []
            for i in range(128):
                acc = workspace[0, i]
                for j in range(1, group):
                    acc = fp32(acc + workspace[j, i])
                reduced.append(acc)
            owned = ordered_sum(parts)
            assert reduced == owned
            assert pullback(x, r, reduced) == pullback(x, r, owned)
            narrow = pullback(x, r, owned, rounded_y=True)
            if narrow != pullback(x, r, owned):
                raw_rounding_witness = {"group": group, "recipe": recipe}
            fp32_stage_cases += 1
    assert raw_rounding_witness is not None
    # Casting each incoming partial before fan-in loses a surviving half-ULP residual.
    incoming = [fp32(1 + 2**-8), fp32(-1)]
    assert bf16(fp32(sum(incoming))) != bf16(fp32(sum(bf16(v) for v in incoming)))
    assert ordered_sum([[2**24], [1.0], [-(2**24)]]) != ordered_sum([[2**24], [-(2**24)], [1.0]])
    values = [0.0] * 128
    values[0], values[32], values[64], values[96] = 2**24, 1.0, -(2**24), 1.0
    dot = source_dot(values, [1.0] * 128)
    alternate = fp32(fp32(values[0] + values[64]) + fp32(values[32] + values[96]))
    assert dot != alternate  # Changing the quadrant merge tree changes the dot.
    result = {
        "passed": True,
        "native_execution": False,
        "exact_shared_jacobian_cases": exact_cases,
        "fp32_materialized_vs_owned_cases": fp32_stage_cases,
        "rounded_forward_row_counterexample": raw_rounding_witness,
        "counterexamples": [
            "early BF16 partial cast",
            "consumer reorder",
            "channel-dot regrouping",
        ],
        "open": ["native TMEM pipeline and stage equivalence", "saved-r generation lifetime"],
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
