"Recurrent tape retention/replay versus independent exact polynomial gradients."

import json
from fractions import Fraction as F
from pathlib import Path


# record=(q,k,v,alpha,beta); scalar delta state retains the feedback dependency.
def forward(seed, records):
    state = seed
    tape = []
    for _q, k, v, alpha, beta in records:
        before = state
        decayed = alpha * state
        correction = beta * (v - decayed * k)
        state = decayed + correction * k
        tape.append((before, state, correction))
    return tape, state


def affine_boundaries(seed, fragments):
    # Independent state-summary route to recreate nonlocal entry states.
    boundaries = []
    state = seed
    for fragment in fragments:
        boundaries.append(state)
        linear, offset = F(1), F(0)
        for _, key, value, alpha, beta in fragment:
            a = alpha * (1 - beta * key * key)
            b = beta * value * key
            linear, offset = linear * a, offset * a + b
        state = state * linear + offset
    return boundaries


def gradients(records, tape, weights, terminal):
    result = [[F(0)] * 5 for _ in records]
    adjoint = terminal
    for t in reversed(range(len(records))):
        q, k, v, alpha, beta = records[t]
        before, after, _ = tape[t]
        result[t][0] = weights[t] * after
        adjoint += weights[t] * q
        result[t][1] = adjoint * (beta * v - 2 * alpha * beta * before * k)
        result[t][2] = adjoint * beta * k
        result[t][3] = adjoint * before * (1 - beta * k * k)
        result[t][4] = adjoint * (v * k - alpha * before * k * k)
        adjoint *= alpha * (1 - beta * k * k)
    return result, adjoint


def loss(seed, records, weights, terminal):
    # Independent expanded affine equation, no cached tape or correction helper.
    state, value = seed, F(0)
    for (q, k, v, alpha, beta), weight in zip(records, weights, strict=True):
        state = alpha * (1 - beta * k * k) * state + beta * v * k
        value += weight * q * state
    return value + terminal * state


if __name__ == "__main__":
    replay_cases = gradient_entries = 0
    for lengths in [[2, 3, 1], [1, 1, 2], [0, 2, 0, 3], [3, 1, 2, 1]]:
        for recipe in range(4):
            records = [
                (
                    F(1 + t % 2, 3),
                    F((t + recipe) % 3 - 1, 2),
                    F(t - recipe, 3),
                    F(1, 2) if t % 2 else F(1),
                    F((t + recipe) % 3, 3),
                )
                for t in range(sum(lengths))
            ]
            seed, terminal = F(1 + recipe, 2), F(1, 3)
            weights = [F(t % 3 - 1, 2) for t in range(len(records))]
            full, _ = forward(seed, records)
            fragments, offset, saved_entries = [], 0, []
            for length in lengths:
                saved_entries.append(full[offset][0] if offset < len(full) else full[-1][1])
                fragments.append(records[offset : offset + length])
                offset += length
            recreated = affine_boundaries(seed, fragments)
            assert recreated == saved_entries
            by_mode = {}
            by_mode["full_tape"] = full
            for mode, entries in [("entry_boundaries", saved_entries), ("nothing", recreated)]:
                replayed = []
                for entry, fragment in zip(entries, fragments, strict=True):
                    local, _ = forward(entry, fragment)
                    replayed.extend(local)
                assert replayed == full
                by_mode[mode] = replayed
            expected_grad, seed_grad = gradients(records, full, weights, terminal)
            for tape in by_mode.values():
                assert gradients(records, tape, weights, terminal) == (expected_grad, seed_grad)
                replay_cases += 1
            # Each parameter appears at most quadratically in this scalar polynomial,
            # so symmetric finite differences are exact, not a small-epsilon approximation.
            step = F(1, 7)
            for t in range(len(records)):
                for parameter in range(5):
                    plus, minus = [list(r) for r in records], [list(r) for r in records]
                    plus[t][parameter] += step
                    minus[t][parameter] -= step
                    derivative = (
                        loss(seed, plus, weights, terminal) - loss(seed, minus, weights, terminal)
                    ) / (2 * step)
                    assert derivative == expected_grad[t][parameter]
                    gradient_entries += 1
            derivative_seed = (
                loss(seed + step, records, weights, terminal)
                - loss(seed - step, records, weights, terminal)
            ) / (2 * step)
            assert derivative_seed == seed_grad
    # An empty local tape is an identity; external exchange remains required.
    for _mode in ["full_tape", "entry_boundaries", "nothing"]:
        empty_tape, unchanged = forward(F(3), [])
        assert empty_tape == [] and unchanged == 3
        assert gradients([], [], [], F(2)) == ([], F(2))
        replay_cases += 1
    records = [(F(1), F(1, 2), F(2), F(1, 2), F(1, 3)), (F(2), F(-1, 2), F(1), F(1), F(1, 2))]
    tape, state = forward(F(3), records)
    bad, _ = forward(F(0), records)
    assert bad != tape  # A crossing fragment cannot replay from rank-local zero.
    mutated = [records[0], (F(2), F(1), F(1), F(1), F(1, 2))]
    assert forward(F(3), mutated)[0] != tape  # Stale inputs with retained boundaries.
    assert (
        gradients(records, tape, [F(1), F(1)], F(1))[0]
        != gradients(records, tape, [F(1), F(1)], F(0))[0]
    )  # Future boundary adjoint is still needed.
    result = {
        "passed": True,
        "native_execution": False,
        "retention_replay_cases": replay_cases,
        "exact_polynomial_gradient_entries": gradient_entries,
        "oracle": "Independent scalar affine loss and exact symmetric polynomial derivatives",
        "counterexamples": [
            "zero instead of crossing entry seed",
            "mutated replay inputs",
            "missing reverse boundary adjoint",
        ],
        "open": [
            "native factor/tape stage bit equality",
            "cross-rank exchange order/progress",
            "checkpoint frame liveness and last-reader storage release",
        ],
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
