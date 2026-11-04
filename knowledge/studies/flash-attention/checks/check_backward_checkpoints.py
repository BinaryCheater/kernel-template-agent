"Logical checkpoint witnesses, not native CUDA execution or accuracy certification."

import json
import math
import struct
from pathlib import Path


def fp32(value):
    return struct.unpack(">f", struct.pack(">f", value))[0]


def bf16_rne(value):
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack(">f", struct.pack(">I", rounded))[0]


def checkpoint(value):
    high = bf16_rne(value)
    residual = fp32(value - high)
    low = bf16_rne(residual)
    restored = fp32(high + low)
    bound = abs((value - high) - residual) + abs(residual - low) + abs((high + low) - restored)
    assert abs(value - restored) <= bound
    return high, low, restored


def loss(logits, values, output_gradient, lse_gradient):
    maximum = max(logits)
    weights = [math.exp(z - maximum) for z in logits]
    total = sum(weights)
    probabilities = [w / total for w in weights]
    output = sum(p * v for p, v in zip(probabilities, values, strict=True))
    lse = maximum + math.log(total)
    return output_gradient * output + lse_gradient * lse, probabilities, output


def derivative_case(logits, values, incoming_lse):
    incoming_output = 2.0
    _, probabilities, output = loss(logits, values, incoming_output, incoming_lse)
    center = incoming_output * output - incoming_lse
    analytic = [
        p * (incoming_output * v - center) for p, v in zip(probabilities, values, strict=True)
    ]
    epsilon = 1e-5
    for j, expected in enumerate(analytic):
        plus, minus = list(logits), list(logits)
        plus[j] += epsilon
        minus[j] -= epsilon
        observed = (
            loss(plus, values, incoming_output, incoming_lse)[0]
            - loss(minus, values, incoming_output, incoming_lse)[0]
        ) / (2 * epsilon)
        assert math.isclose(expected, observed, rel_tol=1e-8, abs_tol=1e-8)
    assert math.isclose(sum(analytic), incoming_lse, abs_tol=1e-12)
    if incoming_lse:
        wrong_center = incoming_output * output
        wrong = [
            p * (incoming_output * v - wrong_center)
            for p, v in zip(probabilities, values, strict=True)
        ]
        assert not math.isclose(sum(wrong), incoming_lse, abs_tol=1e-12)
    return {"incoming_natural_lse_gradient": incoming_lse, "logit_gradients": analytic}


if __name__ == "__main__":
    exact = fp32(1 + 2**-9)
    high, low, restored = checkpoint(exact)
    assert restored == exact and high != exact
    assert bf16_rne(high + low) != exact
    inexact = fp32(1 + 2**-9 + 2**-18)
    high2, low2, restored2 = checkpoint(inexact)
    assert restored2 != inexact
    assert abs(inexact - restored2) < abs(inexact - high2)
    cases = []
    for sink in [False, True]:
        # The sink has zero value but belongs to the normalization and derivative sum.
        logits = [math.log(1), math.log(2 if sink else 3)]
        values = [2.0, 6.0]
        if sink:
            logits.append(math.log(1))
            values.append(0.0)
        for incoming_lse in [0.0, 0.5]:
            cases.append({"sink": sink, **derivative_case(logits, values, incoming_lse)})
    result = {
        "kind": "logical_rounding_residual_and_softmax_derivative_checkpoints",
        "command": (
            ".venv/bin/python knowledge/studies/flash-attention/checks/check_b"
            "ackward_checkpoints.py"
        ),
        "native_execution": False,
        "rounding_policy": "Explicit FP32 arithmetic and finite normal BF16 nearest-even examples",
        "residual_examples": [
            {"input": exact, "high": high, "low": low, "restored": restored},
            {"input": inexact, "high": high2, "low": low2, "restored": restored2},
        ],
        "low_precision_add_loses_residual": True,
        "pair_is_not_lossless": True,
        "gradient_cases": cases,
        "finite_difference_epsilon": 1e-5,
        "finite_difference_absolute_tolerance": 1e-8,
        "remaining": (
            "Native cast/subnormal policies, checkpoint and derivative accurac"
            "y, row visibility, and global synchronization"
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
