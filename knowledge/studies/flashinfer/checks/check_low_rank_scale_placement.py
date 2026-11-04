"""Logical raw/folded adapter and smoothing-channel witnesses; no native GEMM."""

import json
import math
import struct
from pathlib import Path


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bf16(x):
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


if __name__ == "__main__":
    alpha, gain = 3.0, 1.0
    base, bias = 0.125, -0.25
    down, up = [2.0, -1.0], [1.0, 1.5]
    ideal_fold = [gain / alpha * u for u in up]
    ideal = alpha * (base + dot(down, ideal_fold)) + bias
    raw = alpha * base + gain * dot(down, up) + bias
    assert math.isclose(ideal, raw)
    ratio = f32(gain / alpha)
    folded = [bf16(f32(u * ratio)) for u in up]
    prepared = alpha * (base + dot(down, folded)) + bias
    assert bf16(prepared) != bf16(raw)
    # A different final rounding bin can hide that same fold error.
    assert bf16(prepared + 0.5) == bf16(raw + 0.5)
    factor_error = [uf - gain / alpha * u for uf, u in zip(folded, up, strict=True)]
    bound = abs(alpha) * sum(abs(d) * abs(e) for d, e in zip(down, factor_error, strict=True))
    assert abs(prepared - raw) <= bound + 1e-15
    assert alpha * (base + dot(down, folded) + bias) != prepared
    assert alpha * prepared != prepared
    # Raw alpha0 remains a valid correction; no folded ratio exists.
    assert 0 * base + gain * dot(down, up) + bias == 0.25
    # Same-shape changed parameters do not make an old prepared factor fresh.
    changed_up = [2.0, 1.5]
    refolded = [bf16(f32(u * ratio)) for u in changed_up]
    assert alpha * (base + dot(down, refolded)) + bias != prepared
    new_gain = 2.0
    new_folded = [bf16(f32(u * f32(new_gain / alpha))) for u in up]
    assert alpha * (base + dot(down, new_folded)) + bias != prepared
    assert [bf16(f32(u * 0)) for u in up] == [0.0, 0.0]
    x, smooth, l2 = [1.0, 2.0], [2.0, 3.0], [1.0, 3.0]
    smoothed_l2 = [s * w for s, w in zip(smooth, l2, strict=True)]
    smoothed_x = [v * s for v, s in zip(x, smooth, strict=True)]
    assert dot(x, smoothed_l2) == dot(smoothed_x, l2) == 20
    assert dot(smoothed_x, smoothed_l2) != 20
    assert dot([2.0, 8.0], l2) != 20
    assert bf16(1.003 - 0.999) != bf16(bf16(1.003) + bf16(-0.999))
    # E4M3 scale values embed exactly in BF16. The native direct/fallback
    # W4A16 decoder paths can therefore use the same scale for finite codes.
    decoder_cases = 0
    for raw_scale in range(127):
        exponent, mantissa = divmod(raw_scale, 8)
        scale = mantissa * 2**-9 if exponent == 0 else (1 + mantissa / 8) * 2 ** (exponent - 7)
        assert bf16(scale) == scale
        for code in (-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6):
            assert bf16(bf16(code) * bf16(scale)) == bf16(f32(code * scale))
            decoder_cases += 1
    result = {
        "kind": "logical_low_rank_scale_check",
        "native_execution": False,
        "status": "passed",
        "raw_output_before_cast": raw,
        "prepared_output_before_cast": prepared,
        "fold_error_bound": bound,
        "finite_decoder_cases": decoder_cases,
        "checks": [
            "real_scale_identity",
            "bf16_fold_error",
            "bias_after_scale",
            "common_scale_once",
            "alpha_zero_raw_only",
            "changed_weight_refold",
            "changed_scalar_refold",
            "zero_adapter",
            "original_down_input",
            "no_double_smoothing",
            "unfused_intermediate_rounding",
            "finite_w4a16_decoder_equivalence",
        ],
        "numeric_boundary": (
            "Finite codebook and Python/FP32/BF16 witnesses; native MMA, FTZ a"
            "nd concurrency are not replayed."
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
