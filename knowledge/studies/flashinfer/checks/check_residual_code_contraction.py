"""Logical E4M3 component contraction witnesses; no native MMA replay."""

import json
import math
import struct
from pathlib import Path


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def codebook():
    entries = []
    for raw in range(127):
        exponent, mantissa = divmod(raw, 8)
        value = mantissa * 2**-9 if exponent == 0 else (1 + mantissa / 8) * 2 ** (exponent - 7)
        entries.append((value, raw))
        if value:
            entries.append((-value, raw | 128))
    return entries


def cast(x):
    value, _ = min(codebook(), key=lambda item: (abs(item[0] - x), item[1] & 1))
    return value


def pair(x):
    bounded = max(-448.0, min(448.0, f32(x)))
    high = cast(bounded)
    low = cast(max(-448.0, min(448.0, f32(bounded - high))))
    return high, low


def dot32(xs, ys):
    acc = 0.0
    for x, y in zip(xs, ys, strict=True):
        acc = f32(acc + f32(x * y))
    return acc


if __name__ == "__main__":
    high, low = pair(1.09)
    assert low < 0
    assert abs(high + low - f32(1.09)) < abs(high - f32(1.09))
    assert cast(high + low) == high
    assert high + abs(low) != high + low
    tiny_high, tiny_low = pair(1 + 2**-12)
    assert tiny_high == 1 and tiny_low == 0
    assert tiny_high + tiny_low != f32(1 + 2**-12)
    # Source per-row/tile encode/decode scale; original mass remains separate.
    weights = [0.17, 0.49, 1.0]
    values = [4.0, -3.0, 2.0]
    scale = f32(f32(max(max(weights), 1e-10)) * f32(1 / 448))
    inv = f32(1 / scale)
    codes = [pair(f32(w * inv)) for w in weights]
    high_values, low_values = map(list, zip(*codes, strict=True))
    high_dot = sum(h * v for h, v in zip(high_values, values, strict=True))
    low_dot = sum(lo * v for lo, v in zip(low_values, values, strict=True))
    numerator = scale * (high_dot + low_dot)
    mass = sum(weights)
    output = numerator / mass
    reconstructed = [scale * (h + lo) for h, lo in codes]
    expected = sum(w * v for w, v in zip(reconstructed, values, strict=True)) / mass
    assert math.isclose(output, expected, rel_tol=1e-14)
    assert output != scale * high_dot / mass
    assert output != numerator / sum(reconstructed)
    assert output != numerator / (2 * mass)
    # Actual native MMA differs; this scalar FP32 witness only shows order matters.
    trials = ([1e8, -1e8, 1.0], [1e-7, 1e7, -1e7], [3.25, -7.5, 1e6])
    order_witness = None
    for ys in trials:
        separate = f32(dot32(high_values, ys) + dot32(low_values, ys))
        merged_first = dot32([f32(h + lo) for h, lo in codes], ys)
        if separate != merged_first:
            order_witness = {"values": ys, "separate": separate, "merged_first": merged_first}
            break
    assert order_witness is not None
    wrong_pair = (
        scale
        * (high_dot + sum(lo * v for lo, v in zip(reversed(low_values), values, strict=True)))
        / mass
    )
    assert wrong_pair != output
    result = {
        "kind": "logical_residual_code_contraction_check",
        "native_execution": False,
        "status": "passed",
        "signed_residual": [high, low],
        "output": output,
        "scalar_fp32_order_witness": order_witness,
        "checks": [
            "signed_residual",
            "residual_underflow",
            "same_scale_components",
            "original_mass",
            "single_code_projection",
            "component_pairing",
            "narrow_reconstruction_loses_low",
            "contraction_order",
        ],
        "numeric_boundary": (
            "E4M3 finite RN codebook and scalar FP32 witnesses; native packed "
            "MMA association and its error bound remain separate."
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
