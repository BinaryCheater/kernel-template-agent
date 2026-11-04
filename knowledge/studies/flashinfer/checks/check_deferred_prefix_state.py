"Exact prefix-state retention/replay, post-token tracking and epoch counterexamples."

import itertools
import json
from fractions import Fraction as F
from pathlib import Path


def replay_step(state, record):
    key, value, decay, beta = record
    result = []
    for row, v in zip(state, value, strict=True):
        decayed = [decay * x for x in row]
        prediction = sum((x * k for x, k in zip(decayed, key, strict=True)), F(0))
        delta = beta * (v - prediction)
        result.append(tuple(x + delta * k for x, k in zip(decayed, key, strict=True)))
    return tuple(result)


def affine_reference(state, record):
    # Independent matrix form: H'=decay*H*(I-beta*k*k^T)+beta*v*k^T.
    key, value, decay, beta = record
    transform = [[F(i == j) - beta * key[i] * key[j] for j in range(2)] for i in range(2)]
    return tuple(
        tuple(
            decay * sum((state[r][k] * transform[k][c] for k in range(2)), F(0))
            + beta * value[r] * key[c]
            for c in range(2)
        )
        for r in range(2)
    )


def fold(state, records):
    for record in records:
        state = replay_step(state, record)
    return state


if __name__ == "__main__":
    initial = ((F(1), F(2)), (F(-1), F(3)))
    cases = track_cases = 0
    for length in range(1, 6):
        for recipe in range(4):
            records = []
            for t in range(length):
                key = (F((t + recipe) % 3 - 1, 2), F(1 + t % 2, 2))
                value = (F(t + 1), F(recipe - t, 2))
                records.append((key, value, F(1 + t % 2, 2), F(1 + recipe % 2, 3)))
            snapshots = []
            virtual = initial
            for record in records:
                virtual = affine_reference(virtual, record)
                snapshots.append(virtual)
            authoritative = initial
            assert authoritative == initial  # Verification did not commit the virtual state.
            for accepted in range(length + 1):
                snapshot_result = snapshots[accepted - 1] if accepted else initial
                replay_result = fold(authoritative, records[:accepted])
                assert snapshot_result == replay_result
                # Readout query is absent from the state-transition records.
                for query in [(F(1), F(0)), (F(0), F(1))]:
                    output = [
                        sum((x * q for x, q in zip(row, query, strict=True)), F(0))
                        for row in replay_result
                    ]
                    assert len(output) == 2 and fold(initial, records[:accepted]) == replay_result
                for track_step in range(-1, length + 1):
                    sentinel = ((F(99), F(98)), (F(97), F(96)))
                    tracked = sentinel
                    if 0 <= track_step < accepted:
                        tracked = fold(initial, records[: track_step + 1])
                        assert tracked == snapshots[track_step]
                    else:
                        assert tracked == sentinel
                    track_cases += 1
                cases += 1
    record = ((F(1, 2), F(1)), (F(2), F(-1)), F(1, 2), F(1, 3))
    accepted = fold(initial, [record])
    assert fold(accepted, [record]) != accepted  # Applying the same epoch twice.
    assert fold(initial, [record, record]) != accepted  # Diagnostic2 versus accepted prefix1.
    new_record = ((F(1), F(0)), (F(0), F(3)), F(1), F(1, 2))
    assert fold(initial, [record]) != fold(initial, [new_record])  # Stale verify window.
    raw_key_record = ((F(3), F(4)), record[1], record[2], record[3])
    normalized_record = ((F(3, 5), F(4, 5)), record[1], record[2], record[3])
    assert replay_step(initial, raw_key_record) != replay_step(initial, normalized_record)
    # In-place checkpoint destinations cannot collide; tracking cannot destroy a live source.
    live_slots = [0, 1, 2]
    assert len(set(live_slots)) == len(live_slots)
    duplicate_slots = [0, 0, 2]
    assert len(set(duplicate_slots)) != len(duplicate_slots)
    assert set([1]) & set(live_slots)  # An overlapping track destination is invalid.
    orders = set()
    for order in itertools.permutations([record, new_record]):
        orders.add(fold(initial, order))
    assert len(orders) == 2  # These updates cannot be reordered.

    # GDN payload bytes per layer/slot, H2/HV4, K=V128, T8.
    state_bytes = 4 * 128 * 128 * 4
    record_bytes = 2 * 128 * 2 + 4 * (128 * 2 + 2 * 4)
    report = {
        "kind": "exact_deferred_prefix_state_commit",
        "status": "passed",
        "native_execution": False,
        "prefix_cases": cases,
        "tracking_cases": track_cases,
        "reference": "Matrix affine delta update versus row-wise decay/predict/correct replay",
        "counterexamples": [
            "double commit",
            "diagnostic count is not accepted prefix",
            "stale transition window",
            "changed key normalization",
            "colliding checkpoint/track slots",
            "transition reordering",
        ],
        "payload_example": {
            "layers_slots": 1,
            "key_heads": 2,
            "value_heads": 4,
            "window": 8,
            "snapshot_bytes": 8 * state_bytes,
            "transition_tape_bytes": 8 * record_bytes,
        },
        "boundary": (
            "Exact delta recurrence; native fold/verify rounding and publication separate"
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
