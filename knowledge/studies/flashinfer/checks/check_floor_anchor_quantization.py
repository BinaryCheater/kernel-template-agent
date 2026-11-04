"Logical MX floor anchors, exponent reciprocals and complete-group witnesses."

import json
import math
import struct
from pathlib import Path


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def from_bits(value):
    return struct.unpack("<f", struct.pack("<I", value))[0]


def f32(value):
    return from_bits(bits(value))


def source_scale(maximum, anchor_exponent):
    truncated = from_bits(bits(maximum) & 0x7F800000)
    scale = f32(truncated * 2.0**-anchor_exponent)
    reciprocal = from_bits(0x7F000000 - bits(scale))
    if scale == 0.0:
        reciprocal = 0.0
    return scale, reciprocal


def cast_rn_satfinite(value, levels):
    # Ordered code parity encodes the least-significant significand bit for ties.
    index = min(range(len(levels)), key=lambda i: (abs(levels[i] - abs(value)), i % 2))
    return math.copysign(levels[index], value)


if __name__ == "__main__":
    cases = {}
    for name, anchor_exponent, minimum in [("mxfp8", 8, -118), ("mxfp4", 2, -124)]:
        count = 0
        for exponent in range(minimum, 128):
            for mantissa in [1.0, 1.25, 1.5, 1.75, 2.0 - 2.0**-23]:
                maximum = f32(mantissa * 2.0**exponent)
                scale, reciprocal = source_scale(maximum, anchor_exponent)
                assert scale == 2.0 ** (exponent - anchor_exponent)
                assert scale * reciprocal == 1.0
                byte = (bits(scale) >> 23) & 255
                assert 1 <= byte <= 254
                assert 2.0 ** (byte - 127) == scale
                count += 1
        cases[name] = count

    fp8 = [i * 2.0**-9 for i in range(8)]
    fp8 += [
        (1 + m / 8) * 2.0**e for e in range(-6, 9) for m in range(8) if (1 + m / 8) * 2.0**e <= 448
    ]
    fp4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    witnesses = []
    for name, maximum, anchor, largest, levels in [
        ("mxfp8", 500.0, 8, 448.0, fp8),
        ("mxfp4", 7.0, 2, 6.0, fp4),
    ]:
        scale, reciprocal = source_scale(maximum, anchor)
        ceiling = 2.0 ** math.ceil(math.log2(maximum / largest))
        floor_code = cast_rn_satfinite(f32(maximum * reciprocal), levels)
        ceiling_code = cast_rn_satfinite(f32(maximum / ceiling), levels)
        assert scale == 1.0 and ceiling == 2.0
        assert floor_code != ceiling_code
        assert scale * floor_code != ceiling * ceiling_code
        assert cast_rn_satfinite(-maximum * reciprocal, levels) == -floor_code
        witnesses.append(
            {
                "format": name,
                "maximum": maximum,
                "floor_scale": scale,
                "ceiling_scale": ceiling,
                "floor_reconstructed": scale * floor_code,
                "ceiling_reconstructed": ceiling * ceiling_code,
            }
        )

    # One exponent below the admitted input range gives a subnormal scale.
    excluded = []
    for name, anchor, exponent in [("mxfp8", 8, -119), ("mxfp4", 2, -125)]:
        scale, reciprocal = source_scale(2.0**exponent, anchor)
        assert scale == 2.0**-127
        assert reciprocal != 1.0 / scale
        excluded.append(
            {"format": name, "scale": scale, "s_times_bit_reciprocal": scale * reciprocal}
        )
    assert source_scale(0.0, 8) == (0.0, 0.0)
    # Exponent truncation drops a nonzero subnormal maximum entirely.
    assert source_scale(2.0**-149, 8) == (0.0, 0.0)
    assert from_bits(0x7F000000 - bits(2.0**127)) == 0.0

    halves = [[1.0] * 16, [500.0] + [2.0] * 15]
    complete = max(map(max, halves))
    scale, reciprocal = source_scale(complete, 8)
    assert scale == 1.0 and reciprocal == 1.0
    wrong_half_scale, _ = source_scale(max(halves[0]), 8)
    assert wrong_half_scale == 1.0 / 256 and wrong_half_scale != scale
    # The global scale participates in both the maximum and code multiplier.
    group = [0.25, -0.5, 1.0] + [0.0] * 29
    global_scale = 8.0
    scale, reciprocal = source_scale(f32(max(map(abs, group)) * global_scale), 8)
    multiplier = f32(reciprocal * global_scale)
    assert scale == 1.0 / 32 and multiplier == 256.0
    codes = [cast_rn_satfinite(f32(x * multiplier), fp8) for x in group]
    assert [scale * q for q in codes[:3]] == [2.0, -4.0, 8.0]
    assert scale * cast_rn_satfinite(group[2] * reciprocal, fp8) != 8.0

    output = {
        "kind": "logical_floor_anchor_quantization",
        "native_execution": False,
        "status": "passed",
        "normal_maximum_cases": cases,
        "floor_ceiling_witnesses": witnesses,
        "excluded_subnormal_scales": excluded,
        "zero_multiplier_branch": "checked; stored UE8M0 zero conversion not claimed",
        "group_maximum": "paired M16 pieces recover complete M32 maximum",
        "global_scale": "max and multiplier both required",
        "boundary": "RN-SATFINITE logical casts; no native scalar/NVVM lowering claim",
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(output, indent=2) + "\n")
