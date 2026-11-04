"Logical rematerialization and vocabulary-partition witnesses; no native execution."

import json
import math
from pathlib import Path


def logsumexp(values):
    maximum = max(values)
    return maximum + math.log(sum(math.exp(x - maximum) for x in values))


def full_loss(logits, target, scale, smoothing, penalty):
    scaled = [scale * x for x in logits]
    normalizer = logsumexp(scaled)
    loss = (
        normalizer
        - (1 - smoothing) * scaled[target]
        - smoothing / len(logits) * sum(scaled)
        + penalty * normalizer**2
    )
    return loss, normalizer


def partitioned_loss(logits, target, scale, smoothing, penalty):
    scaled = [scale * x for x in logits]
    shards = [scaled[:2], scaled[2:]]
    local_lse, additive = [], []
    for rank, shard in enumerate(shards):
        local_lse.append(logsumexp(shard))
        value = -smoothing / len(logits) * sum(shard)
        local_target = target - rank * 2
        if 0 <= local_target < len(shard):
            value -= (1 - smoothing) * shard[local_target]
        additive.append(value)
    global_lse = logsumexp(local_lse)
    return global_lse + sum(additive) + penalty * global_lse**2, global_lse


if __name__ == "__main__":
    inputs, weights, bias = [1.0, 2.0], [[2.0, -1.0], [-1.0, 2.0]], [1.0, -4.0]

    def affine(w):
        return [
            sum(x * y for x, y in zip(inputs, row, strict=True)) + b
            for row, b in zip(w, bias, strict=True)
        ]

    def activation(values):
        return [max(x, 0.0) ** 2 for x in values]

    a = affine(weights)
    b = activation(a)
    both_values = (a, b)
    affine_only = (a, activation(a))
    replay_a = affine(weights)
    earlier_boundary = (replay_a, activation(replay_a))
    assert both_values == affine_only == earlier_boundary
    assert affine([[3.0, -1.0], [-1.0, 2.0]]) != a
    assert [1.0 > 0, -1.0 > 0] == [2.0 > 0, -2.0 > 0]
    assert [max(1.0, 0), max(-1.0, 0)] != [max(2.0, 0), max(-2.0, 0)]

    logits, scale = [0.0, math.log(2), math.log(3), math.log(4)], 0.7
    cases = []
    for target in [0, 3]:
        for smoothing in [0.0, 0.2]:
            for penalty in [0.0, 0.03]:
                full, lse = full_loss(logits, target, scale, smoothing, penalty)
                split, split_lse = partitioned_loss(logits, target, scale, smoothing, penalty)
                assert math.isclose(full, split, abs_tol=1e-12)
                assert math.isclose(lse, split_lse, abs_tol=1e-12)
                gradients = [
                    scale
                    * (
                        (1 + 2 * penalty * lse) * math.exp(scale * x - lse)
                        - smoothing / len(logits)
                        - (1 - smoothing) * (j == target)
                    )
                    for j, x in enumerate(logits)
                ]
                epsilon = 1e-5
                for j, expected in enumerate(gradients):
                    plus, minus = list(logits), list(logits)
                    plus[j] += epsilon
                    minus[j] -= epsilon
                    observed = (
                        partitioned_loss(plus, target, scale, smoothing, penalty)[0]
                        - partitioned_loss(minus, target, scale, smoothing, penalty)[0]
                    ) / (2 * epsilon)
                    assert math.isclose(observed, expected, abs_tol=1e-8)
                cases.append(
                    {
                        "target": target,
                        "smoothing": smoothing,
                        "squared_lse_scale": penalty,
                        "result": "passed",
                    }
                )
    wrong_lse = logsumexp([lse, lse])
    assert math.isclose(wrong_lse - lse, math.log(2), abs_tol=1e-12)
    result = {
        "kind": "logical_auxiliary_retention_and_partitioned_loss",
        "native_execution": False,
        "command": (
            ".venv/bin/python knowledge/studies/flash-attention/checks/check_auxiliary_choices.py"
        ),
        "retention_modes": "same affine and squared-ReLU values for three boundaries",
        "changed_weights_break_replay": True,
        "relu_bitmask_cannot_recover_output": True,
        "loss_cases": cases,
        "global_lse_reused_as_local_error": "log(2) for two shards",
        "finite_difference_epsilon": epsilon,
        "remaining": (
            "Native replay precision, distributed row identity/completion and output casts"
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
