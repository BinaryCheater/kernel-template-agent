"""Logical common-key translation and quantizer witnesses, not GPU attention."""

import json
import math
from pathlib import Path


def softmax(scores):
    m = max(scores)
    weights = [math.exp(x - m) for x in scores]
    return [x / sum(weights) for x in weights], m + math.log(sum(weights))


def quantize(xs):
    scale = max(max(abs(x) for x in xs), 1e-12) / 127
    return [min(127, max(-127, round(x / scale))) * scale for x in xs]


if __name__ == "__main__":
    keys = [1024.0, 1032.0, 1040.0, 1048.0]
    mu = sum(keys) / len(keys)
    centered = [k - mu for k in keys]
    rows = []
    for q, support in [(0.125, [0, 1, 2, 3]), (-0.25, [0, 2]), (0.5, [1, 3])]:
        shift = q * mu
        original, lse = softmax([q * keys[j] for j in support])
        translated, centered_lse = softmax([q * centered[j] for j in support])
        assert all(
            math.isclose(a, b, abs_tol=1e-14) for a, b in zip(original, translated, strict=True)
        )
        assert math.isclose(centered_lse + shift, lse, abs_tol=1e-12)
        assert not math.isclose(centered_lse, lse)
        rows.append(dict(query=q, support=support, restored_lse=lse, row_shift=shift))
    base, _ = softmax([0.125 * k for k in keys])
    per_block = [k - sum(keys[j // 2 * 2 : j // 2 * 2 + 2]) / 2 for j, k in enumerate(keys)]
    wrong, _ = softmax([0.125 * k for k in per_block])
    assert any(abs(a - b) > 0.01 for a, b in zip(base, wrong, strict=True))
    # A denominator-only logit must shift along with ordinary logits.
    ordinary = [1.0, 2.0]
    original_sink, _ = softmax([*ordinary, 0.0])
    shifted_sink, _ = softmax([x - 3 for x in ordinary] + [-3.0])
    wrong_sink, _ = softmax([x - 3 for x in ordinary] + [0.0])
    assert all(math.isclose(a, b) for a, b in zip(original_sink, shifted_sink, strict=True))
    assert wrong_sink != original_sink
    uncentered_decoded = quantize(keys)
    centered_decoded = [x + mu for x in quantize(centered)]
    error_raw = max(abs(a - b) for a, b in zip(keys, uncentered_decoded, strict=True))
    error_centered = max(abs(a - b) for a, b in zip(keys, centered_decoded, strict=True))
    assert error_centered < error_raw and uncentered_decoded != centered_decoded
    result = {
        "kind": "logical_softmax_key_centering",
        "native_execution": False,
        "command": ".venv/bin/python knowledge/studies/flashinfer/checks/check_key_centering.py",
        "key_mean": mu,
        "row_cases": rows,
        "wrong_per_block_center_changes_weights": True,
        "unshifted_sink_changes_weights": True,
        "int8_max_error_uncentered": error_raw,
        "int8_max_error_centered": error_centered,
        "remaining": (
            "Native FP32 mean/subtraction, source 128-row/64-key quantizer gro"
            "ups, MMA/exp arithmetic and admitted end-to-end error"
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
