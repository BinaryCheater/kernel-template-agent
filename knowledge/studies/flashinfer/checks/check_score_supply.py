"Logical two-owner score supply and rounded-mass checks, not device execution."

import json
import math
import struct
from pathlib import Path


def half(x):
    return struct.unpack("<e", struct.pack("<e", x))[0]


def tile(state, scores, values):
    old_m, old_mass, old_o = state
    new_m = max(old_m, max(scores))
    correction = 0.0 if old_m == -math.inf else 2 ** (old_m - new_m)
    p = [half(2 ** (s - new_m)) for s in scores]
    mass = old_mass * correction + sum(p)
    out = [
        x * correction + sum(w * v[d] for w, v in zip(p, values, strict=True))
        for d, x in enumerate(old_o)
    ]
    return (new_m, mass, out), p, correction


if __name__ == "__main__":
    # Each full score has distinct latent and rope contributions before exponentiation.
    latent_scores = [0.0, 1.0, -0.75, 0.3, 2.5, 0.0, 1.25, -0.4]
    rope_scores = [0.25, -1.5, 0.0, 0.45, 0.0, -0.5, -1.0, 0.0]
    scores = [x + y for x, y in zip(latent_scores, rope_scores, strict=True)]
    values = [[100 * (k + 1) + d for d in range(8)] for k in range(8)]
    state = (-math.inf, 0.0, [0.0] * 8)
    shard_mass = [0.0, 0.0]
    shard_outputs = [[0.0] * 4 for _ in range(2)]
    all_p = []
    for start in [0, 4]:
        old_state = state
        state, p, correction = tile(state, scores[start : start + 4], values[start : start + 4])
        # Each owner finds a key-half maximum; the exchange yields the complete frame.
        maxima = [max(old_state[0], *scores[start + j * 2 : start + j * 2 + 2]) for j in range(2)]
        assert max(maxima) == state[0]
        exchange = [None] * 4
        for owner in range(2):
            local_p = [
                half(2 ** (s - state[0])) for s in scores[start + owner * 2 : start + owner * 2 + 2]
            ]
            exchange[owner * 2 : owner * 2 + 2] = local_p
            shard_mass[owner] = shard_mass[owner] * correction + sum(local_p)
        assert exchange == p
        for owner in range(2):
            for d in range(4):
                shard_outputs[owner][d] = shard_outputs[owner][d] * correction + sum(
                    w * values[start + k][owner * 4 + d] for k, w in enumerate(exchange)
                )
        all_p.append(p)
    complete_mass = sum(shard_mass)
    assert math.isclose(complete_mass, state[1], rel_tol=1e-15)
    outputs = [x / complete_mass for row in shard_outputs for x in row]
    replicated = [x / state[1] for x in state[2]]
    assert outputs == replicated
    duplicate_denominator = [x / (2 * state[1]) for x in state[2]]
    assert all(x == y / 2 for x, y in zip(duplicate_denominator, outputs, strict=True))
    # Neither feature owner may read only its former key shard.
    wrong = sum(all_p[1][k] * values[4 + k][0] for k in range(2)) / complete_mass
    assert wrong != outputs[0]
    last_weights = [2 ** (s - state[0]) for s in scores[4:]]
    assert sum(last_weights) != sum(all_p[1])
    # Softmax must follow the addition of score components, not each component separately.
    latent_p = [2**x for x in latent_scores[:4]]
    rope_p = [2**x for x in rope_scores[:4]]
    separately_added = [
        a / sum(latent_p) + b / sum(rope_p) for a, b in zip(latent_p, rope_p, strict=True)
    ]
    correct = [2**x for x in scores[:4]]
    correct = [x / sum(correct) for x in correct]
    assert any(abs(x / 2 - y) > 0.01 for x, y in zip(separately_added, correct, strict=True))
    result = {
        "kind": "logical_two_owner_score_replication_exchange",
        "native_execution": False,
        "command": ".venv/bin/python knowledge/studies/flashinfer/checks/check_score_supply.py",
        "owners": 2,
        "key_tiles": 2,
        "features": 8,
        "reference_after_tiles": state[0],
        "rounded_mass_shards": shard_mass,
        "complete_rounded_mass": complete_mass,
        "normalized_feature_outputs": outputs,
        "replication_equals_exchange": True,
        "duplicate_full_mass_halves_output": True,
        "partial_key_read_differs": True,
        "rounded_mass_differs_from_ungained_exponent_mass": True,
        "separately_normalized_score_components_differ": True,
        "remaining": (
            "Native FP16/BF16 fragment maps, row-sum/reduction arithmetic, bar"
            "riers, cooperative-grid merge and alias lifetimes"
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
