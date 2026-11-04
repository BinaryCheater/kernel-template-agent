"Logical query mean/residual identity and correction range, not native NVFP4."

import json
import math
import struct
from pathlib import Path


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def weights(scores):
    m = max(scores)
    w = [math.exp(s - m) for s in scores]
    return [x / sum(w) for x in w], m + math.log(sum(w))


if __name__ == "__main__":
    queries = [[4.0, 2.0], [8.0, 4.0], [12.0, 0.0]]
    keys = [[0.0, 1.0], [1.0, -1.0], [2.0, 0.0]]
    mean = [sum(row[d] for row in queries) / len(queries) for d in range(2)]
    assert mean == [8.0, 2.0]
    correction = [dot(mean, k) for k in keys]
    for q in queries:
        residual = [a - b for a, b in zip(q, mean, strict=True)]
        original = [dot(q, k) for k in keys]
        residual_scores = [dot(residual, k) for k in keys]
        complete = [r + c for r, c in zip(residual_scores, correction, strict=True)]
        assert complete == original
        assert weights(complete) == weights(original)
        assert weights(residual_scores)[0] != weights(original)[0]
        wrong_scale = [0.25 * r + c for r, c in zip(residual_scores, correction, strict=True)]
        assert weights(wrong_scale)[0] != weights([0.25 * s for s in original])[0]
    mapping = [i // 128 for i in range(130)]
    assert mapping.count(0) == 128 and mapping.count(1) == 2
    large_correction = dot([256.0, 256.0], [256.0, 256.0])
    try:
        struct.pack("<e", large_correction)
    except OverflowError:
        fp16_overflow = True
    else:
        fp16_overflow = False
    assert fp16_overflow and struct.unpack("<f", struct.pack("<f", large_correction))[0] == 131072
    # Query correction restores q dot Kcentered, not q dot original K.
    key_mean = [1.0, 2.0]
    centered_k = [[x - m for x, m in zip(k, key_mean, strict=True)] for k in keys]
    q = queries[0]
    original_p, original_lse = weights([dot(q, k) for k in keys])
    centered_p, centered_lse = weights([dot(q, k) for k in centered_k])
    assert all(math.isclose(a, b) for a, b in zip(original_p, centered_p, strict=True))
    assert math.isclose(centered_lse + dot(q, key_mean), original_lse)
    result = {
        "kind": "logical_query_center_residual_scores",
        "native_execution": False,
        "command": (
            ".venv/bin/python knowledge/studies/flashinfer/checks/check_query_center_correction.py"
        ),
        "shared_correction": correction,
        "queries_checked": len(queries),
        "omitted_and_wrong_scale_corrections_differ": True,
        "query130_block_counts": [128, 2],
        "fp32_correction": large_correction,
        "fp16_correction_overflows": fp16_overflow,
        "independent_key_center_lse_offset": dot(q, key_mean),
        "remaining": (
            "Native residual NVFP4 rounding, GQA/correction TMA mapping, seede"
            "d MMA accumulation, mask tails and original LSE contract"
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
