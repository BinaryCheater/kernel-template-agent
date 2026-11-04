"""Exact checkpoint/journal rebase and modulo-ring accepted-prefix ownership."""

import json
import math
from fractions import Fraction as F
from pathlib import Path

from check_deferred_prefix_state import affine_reference


def step_with_correction(state, record):
    key, value, decay, beta = record
    u = []
    for row, v in zip(state, value, strict=True):
        predicted = decay * sum((x * k for x, k in zip(row, key, strict=True)), F(0))
        u.append(beta * (v - predicted))
    return affine_reference(state, record), tuple(u)


def materialize(base, entries):
    anchor = entries[-1][2] if entries else F(1)
    state = [[anchor * x for x in row] for row in base]
    for key, u, frame in entries:
        for r in range(len(state)):
            for c in range(len(state[0])):
                state[r][c] += anchor / frame * u[r] * key[c]
    return tuple(map(tuple, state))


def record_at(t):
    return (
        (F(t % 3 - 1, 2), F(1, 2)),
        (F(t, 3), F(2 - t, 2)),
        F(1, 2) if t % 2 else F(1),
        F(1 + t % 2, 3),
    )


if __name__ == "__main__":
    initial = ((F(1), F(2)), (F(-1), F(3)))
    numerical = 0
    for live in range(17):
        entries, state, frame = [], initial, F(1)
        for t in range(live):
            record = record_at(t)
            state, u = step_with_correction(state, record)
            frame *= record[2]
            entries.append((record[0], u, frame))
        assert materialize(initial, entries) == state
        # The stored state-dependent u differs from raw values.
        if entries:
            assert entries[0][1] != record_at(0)[1]
        for draft in [4, 8]:
            for threshold in {1, 17 - draft}:
                flush = live >= threshold
                checkpoint = state if flush else initial
                kept = [] if flush else entries[:]
                new_frame = F(1) if flush else frame
                virtual = state
                expected = [state]
                new_entries = []
                for s in range(draft):
                    record = record_at(live + s)
                    virtual, u = step_with_correction(virtual, record)
                    new_frame *= record[2]
                    new_entries.append((record[0], u, new_frame))
                    expected.append(virtual)
                for accepted in range(draft + 1):
                    actual = materialize(checkpoint, kept + new_entries[:accepted])
                    assert actual == expected[accepted]
                    numerical += 1
    address_cases = accepted_cursors = 0
    bad_restart_overlap = 0
    for base in range(32):
        for live in range(17):
            old = {(base + j) % 32 for j in range(live)}
            for draft in [4, 8]:
                append = [(base + live + s) % 32 for s in range(draft)]
                assert len(set(append)) == draft and not old & set(append)
                # Appending at the old base on flush can overwrite a sibling's keys.
                bad_restart_overlap += bool(old & {(base + s) % 32 for s in range(draft)})
                for threshold in range(1, 18 - draft):
                    flush = live >= threshold
                    assert flush or live + draft <= 16
                    for accepted in range(draft + 1):
                        new_base = (base + live) % 32 if flush else base
                        new_len = accepted if flush else live + accepted
                        assert new_len <= 16
                        actual = [(new_base + j) % 32 for j in range(new_len)]
                        expected = (
                            append[:accepted]
                            if flush
                            else ([(base + j) % 32 for j in range(live)] + append[:accepted])
                        )
                        assert actual == expected
                        accepted_cursors += 1
                    address_cases += 1
    owners = 0
    for key_heads in [1, 2, 4]:
        for ratio in range(1, 5):
            value_heads = key_heads * ratio
            selected = [h for h in range(value_heads) if h % ratio == 0]
            assert [h // ratio for h in selected] == list(range(key_heads))
            owners += len(selected)
    assert bad_restart_overlap > 0
    # A stale absolute draft gate after flush decays the folded base twice.
    folded = ((F(2), F(0)), (F(0), F(0)))
    key, u = (F(0), F(0)), (F(0), F(0))
    assert materialize(folded, [(key, u, F(1, 2))]) != materialize(folded, [(key, u, F(1, 4))])
    # All layers must see old cursor metadata through the shared step.
    old_base, old_len = 29, 12
    first_layer_commit = (old_base + old_len) % 32
    assert [(old_base + j) % 32 for j in range(old_len)] != [
        (first_layer_commit + j) % 32 for j in range(old_len)
    ]
    # Predicating gate exponentiation does not sanitize arbitrary U/K garbage.
    assert math.isnan(0.0 * math.inf)
    result = {
        "passed": True,
        "native_execution": False,
        "virtual_fold_prefix_cases": numerical,
        "ring_threshold_cases": address_cases,
        "accepted_cursor_cases": accepted_cursors,
        "elected_shared_key_writers": owners,
        "unsafe_restart_overlap_cases": bad_restart_overlap,
        "oracle": (
            "Sequential affine state versus exact low-rank materialization and cursor lists"
        ),
        "open": [
            "native stored correction/fold rounding",
            "stochastic RNG identity",
            "all-layer publication and slot lifetime",
            "finite inactive ring payload contract",
        ],
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
