"Scalar representation and counter word-lane witnesses; no native sampler proof."

import json
import math
import struct
from pathlib import Path


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


cases = 0
for weight in (0.0001, 0.01, 0.2, 1.0, 3.0, 10000.0):
    for h in (0, 1, 2**16, 2**31, 2**32 - 2, 2**32 - 1):
        U = 1 - (h + 0.5) / 2**32
        a = math.log(U) / weight
        b = math.log(weight) - math.log(-math.log(U))
        assert math.isclose(-math.log(-a), b, rel_tol=1e-14, abs_tol=1e-14)
        cases += 1
lane_cases = 0
for origin in (0, 4, 1024, 8192):
    for part in range(3):
        for local in range(1024):
            token = origin + part * 1024 + local
            quad = (part * 1024 + origin) // 4 + local // 4
            lane = local % 4
            assert 4 * quad + lane == token
            lane_cases += 1
residual_cases = 0
for p in (0.00001, 0.1, 0.25, 0.5, 1.0):
    for q in (0.0, p / 2, p, p * 2):
        expected = max(p - q, 0.0)
        logres = -math.inf if q >= p else math.log(p) + math.log1p(-q / p)
        assert math.isclose(math.exp(logres), expected, rel_tol=1e-14, abs_tol=1e-15)
        residual_cases += 1
# A positive exact difference can vanish after ratio rounding in the source recipe.
ratio = f32(math.exp(-1e-8))
assert ratio == 1.0
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "passed": True,
            "score_cases": cases,
            "aligned_counter_word_cases": lane_cases,
            "residual_cases": residual_cases,
            "near_equal_counterexample": {
                "logq_minus_logp": -1e-8,
                "f32_exp_ratio": ratio,
                "exact_positive_ratio_gap": -math.expm1(-1e-8),
            },
            "scope": (
                "Host real-valued algebra, aligned quad/lane mapping and FP32 roun"
                "ding boundary; no GPU, generator quality or categorical law claim"
            ),
        },
        indent=2,
    )
    + "\n"
)
