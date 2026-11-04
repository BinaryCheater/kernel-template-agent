"Source integer replay and finite dropout algebra witnesses; not CUDA execution."

import json
import math
from pathlib import Path


def philox_words(seed, subsequence, offset, rounds=7):
    mask = (1 << 32) - 1
    key = [seed & mask, seed >> 32]
    counter = [offset & mask, offset >> 32, subsequence & mask, subsequence >> 32]
    for step in range(rounds):
        product0 = 0xD2511F53 * counter[0]
        product1 = 0xCD9E8D57 * counter[2]
        counter = [
            ((product1 >> 32) ^ counter[1] ^ key[0]) & mask,
            product1 & mask,
            ((product0 >> 32) ^ counter[3] ^ key[1]) & mask,
            product0 & mask,
        ]
        if step + 1 < rounds:
            key = [(key[0] + 0x9E3779B9) & mask, (key[1] + 0xBB67AE85) & mask]
    return counter


def replay(seed, base, batch, head, heads, lane, row16, col32):
    offset = base + (batch * heads + head) * 32 + lane
    subsequence = row16 | (col32 << 32)
    words = philox_words(seed, subsequence, offset)
    return b"".join(word.to_bytes(4, "little") for word in words)


def fixed_mask_loss(logits, values, kept, keep_rate, gradient):
    weights = [math.exp(x) for x in logits]
    total = sum(weights)
    return gradient * sum(
        w / total * m * v / keep_rate for w, m, v in zip(weights, kept, values, strict=True)
    )


if __name__ == "__main__":
    seed, base, heads = 17, 64, 3
    coordinates = [(row, col) for row in [0, 2, 5] for col in [0, 2, 4]]
    forward = {c: replay(seed, base, 1, 2, heads, 14, *c) for c in coordinates}
    backward = {c: replay(seed, base, 1, 2, heads, 14, *c) for c in reversed(coordinates)}
    assert forward == backward
    assert philox_words(seed, 2 | (2 << 32), 238) != philox_words(
        seed, 2 | (2 << 32), 238, rounds=10
    )
    current_offsets = {
        base + (b * heads + h) * 32 + lane
        for b in range(2)
        for h in range(heads)
        for lane in range(32)
    }
    next_offsets = {offset + 2 * heads * 32 for offset in current_offsets}
    assert len(current_offsets) == 192 and current_offsets.isdisjoint(next_offsets)
    threshold = math.floor(0.9 * 255)
    kept_bytes = [byte for byte in range(256) if byte <= threshold]
    assert threshold == 229 and len(kept_bytes) == 230
    assert (threshold <= threshold) and not (threshold + 1 <= threshold)

    zero_cases = []
    for keep in [False, True]:
        for value in [0.0, 0.25, 0.75]:
            encoded = value if keep else -value
            assert abs(encoded) == value
            sign_decode = math.copysign(1.0, encoded) > 0
            assert sign_decode == keep
            if value > 0:
                assert (encoded >= 0) == keep
            else:
                zero_cases.append({"keep": keep, "comparison_decodes_keep": encoded >= 0})
            a, center = 4.0, 1.0
            observed = encoded * (a - center if encoded >= 0 else center)
            assert observed == value * (int(keep) * a - center)
    assert max(-0.75, 0.0) != abs(-0.75)

    probabilities, values, kept = [0.25, 0.75], [2.0, 6.0], [1, 0]
    keep_rate, incoming = 0.5, 2.0
    output = sum(p * m * v / keep_rate for p, m, v in zip(probabilities, kept, values, strict=True))
    scaled_center = keep_rate * output * incoming
    signed = [p if m else -p for p, m in zip(probabilities, kept, strict=True)]
    derivatives = [
        c * (incoming * v - scaled_center if c >= 0 else scaled_center) / keep_rate
        for c, v in zip(signed, values, strict=True)
    ]
    assert derivatives == [1.5, -1.5]
    logits = [math.log(p) for p in probabilities]
    epsilon = 1e-5
    for j, expected in enumerate(derivatives):
        plus, minus = list(logits), list(logits)
        plus[j] += epsilon
        minus[j] -= epsilon
        difference = (
            fixed_mask_loss(plus, values, kept, keep_rate, incoming)
            - fixed_mask_loss(minus, values, kept, keep_rate, incoming)
        ) / (2 * epsilon)
        assert math.isclose(difference, expected, abs_tol=1e-8)

    result = {
        "kind": "source_integer_replay_and_signed_dropout_algebra",
        "command": (
            ".venv/bin/python knowledge/studies/flash-attention/checks/check_dropout_replay.py"
        ),
        "native_execution": False,
        "reordered_canonical_blocks": "passed",
        "generator_rounds": 7,
        "changing_to_ten_rounds_changes_output": True,
        "invocation_offset_reservation": "disjoint for tested 2 batches x 3 heads x 32 lanes",
        "threshold": threshold,
        "uniform_byte_keep_rate": len(kept_bytes) / 256,
        "requested_keep_rate": 0.9,
        "signed_zero_cases": zero_cases,
        "fixed_mask_logit_derivatives": derivatives,
        "finite_difference_epsilon": epsilon,
        "remaining": (
            "Native forward/backward lane-component mapping, packed comparison"
            "s, generator reservation/publication and numeric lowering"
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
