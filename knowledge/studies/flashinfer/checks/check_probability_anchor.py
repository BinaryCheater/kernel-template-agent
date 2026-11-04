"Logical probability factorization/cast witnesses; no native instruction replay."

import json
import math
from pathlib import Path


def e4m3_values():
    values = [mantissa * 2**-9 for mantissa in range(8)]
    for exponent in range(1, 16):
        for mantissa in range(8 if exponent < 15 else 7):
            values.append((1 + mantissa / 8) * 2 ** (exponent - 7))
    return values


def nearest(x, values):
    # Witnesses avoid ties; native tie/FTZ/NaN behavior is a separate obligation.
    return min(values, key=lambda y: abs(x - y))


def encode(logits, reference, span):
    codes, scales, mass = [], [], 0.0
    for begin in range(0, len(logits), span):
        block = logits[begin : begin + span]
        valid = [x for x in block if x != -math.inf]
        if not valid:
            codes.extend([0.0] * len(block))
            scales.extend([0.0] * len(block))
            continue
        anchor = max(valid)
        scale = 2 ** (anchor - reference) / 6
        normalized = [6 * 2 ** (x - anchor) if x != -math.inf else 0.0 for x in block]
        mass += scale * sum(normalized)
        codes.extend(nearest(z, [0, 0.5, 1, 1.5, 2, 3, 4, 6]) for z in normalized)
        scales.extend([nearest(scale, e4m3_values())] * len(block))
    return mass, codes, scales


if __name__ == "__main__":
    grouping = {}
    for span in (16, 32, 64, 128):
        logits = [-i / 8 for i in range(121)] + [-math.inf] * 7
        mass, codes, scales = encode(logits, 0.0, span)
        assert math.isclose(mass, sum(2**x for x in logits), rel_tol=1e-14)
        assert codes[121:] == [0] * 7
        assert all(scales[i : i + span] == [scales[i]] * span for i in range(0, 128, span))
        grouping[str(span)] = sum(c * s for c, s in zip(codes, scales, strict=True)) / mass
    assert len(set(grouping.values())) > 1
    mass, codes, scales = encode([0.0] + [-math.inf] * 127, 0.0, 128)
    numerator = sum(c * s for c, s in zip(codes, scales, strict=True))
    assert mass == 1 and codes[0] == 6 and numerator != mass
    constant_v_output = numerator / mass
    assert constant_v_output == 1.03125
    # Encoding with the rounded decoder scale would define a different code recipe.
    ideal_z = 5.1
    rounded_scale = nearest(1 / 6, e4m3_values())
    assert nearest(ideal_z, [0, 0.5, 1, 1.5, 2, 3, 4, 6]) != nearest(
        (ideal_z / 6) / rounded_scale, [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    )
    tiny_mass, tiny_codes, tiny_scales = encode([-12.0] * 128, 0.0, 128)
    assert tiny_mass > 0 and tiny_codes == [6] * 128 and tiny_scales == [0] * 128
    # A later row reference changes old numerator and old mass equally, once.
    old_mass, old_c, old_s = encode([0.0] * 128, 0.0, 128)
    new_mass, new_c, new_s = encode([10.0] * 128, 10.0, 128)
    old_o = sum(c * s for c, s in zip(old_c, old_s, strict=True))
    new_o = sum(c * s for c, s in zip(new_c, new_s, strict=True))
    correction = 2**-10
    output = (old_o * correction + new_o) / (old_mass * correction + new_mass)
    assert output == constant_v_output
    assert (old_o + new_o) / (old_mass * correction + new_mass) != output
    empty = encode([-math.inf] * 128, 0.0, 128)
    assert empty == (0.0, [0.0] * 128, [0.0] * 128)
    assert nearest(2**8 / 6, e4m3_values()) == 44
    result = {
        "kind": "logical_probability_anchor_check",
        "native_execution": False,
        "status": "passed",
        "anchor_span_output_for_constant_v": grouping,
        "unrounded_mass_constant_v_output": constant_v_output,
        "tiny_scale_positive_mass": tiny_mass,
        "checks": [
            "factorization",
            "mask_tail",
            "shared_anchor_scales",
            "cast_order",
            "unrounded_mass",
            "scale_underflow",
            "reference_correction",
            "empty_span",
            "lag8_scale_range",
        ],
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
