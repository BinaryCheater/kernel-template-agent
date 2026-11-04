"""Nested real-token updates versus expanded64 stream and source query maps."""

import json
from fractions import Fraction as F
from pathlib import Path

from check_deferred_prefix_state import affine_reference, replay_step


def readout(state, query):
    return tuple(sum((x * q for x, q in zip(row, query, strict=True)), F(0)) for row in state)


if __name__ == "__main__":
    initial = ((F(1), F(2)), (F(-1), F(3)))
    cases = prefix_cases = phase_cases = 0
    for n in range(1, 5):
        for length in [1, 2, 21, 22, 63, 64, 65]:
            for recipe in range(2):
                gates = [F(1, 2) if t % 2 else F(1) for t in range(length)]
                queries = [(F(t % 3, 3), F(1)) for t in range(length)]
                records = []
                for t in range(length):
                    token = []
                    for j in range(n):
                        key = (F((t + j) % 3 - 1, 2), F(1, 2))
                        value = (F(j - t, 3), F(recipe + j, 2))
                        token.append(
                            (key, value, gates[t] if j == 0 else F(1), F((t + j + recipe) % 3, 3))
                        )
                    records.append(token)
                # Independent nested contract: decay once, then undamped corrections.
                state = initial
                expected_states = []
                expected_outputs = []
                for t, token in enumerate(records):
                    state = tuple(tuple(gates[t] * x for x in row) for row in state)
                    for key, value, _, beta in token:
                        state = replay_step(state, (key, value, F(1), beta))
                    expected_states.append(state)
                    expected_outputs.append(readout(state, queries[t]))
                flat = [record for token in records for record in token]
                expanded_states = []
                expanded_outputs = []
                state = initial
                for start in range(0, len(flat), 64):
                    # A chunk boundary can occur at substep1 or2 for n3.
                    phase_cases += start % n != 0
                    for offset, record in enumerate(flat[start : start + 64]):
                        m = start + offset
                        state = affine_reference(state, record)
                        expanded_states.append(state)
                        if (m + 1) % n == 0:
                            expanded_outputs.append(readout(state, queries[m // n]))
                assert expanded_outputs == expected_outputs
                assert state == expected_states[-1]
                for accepted in range(length + 1):
                    expanded = expanded_states[n * accepted - 1] if accepted else initial
                    nested = expected_states[accepted - 1] if accepted else initial
                    assert expanded == nested
                    prefix_cases += 1
                cases += 1
    # Reconstruct source's cp_* real-readout map from expanded chunk index.
    mapping_cases = masks = 0
    for n in range(1, 5):
        for length in range(1, 131):
            seen = []
            for chunk in range((length * n + 63) // 64):
                block, inner = divmod(chunk, n)
                sub, rem = divmod(inner * 64, n)
                first_row = n - 1 - rem
                count = (63 - first_row) // n + 1
                source = [
                    block * 64 + sub + j for j in range(count) if block * 64 + sub + j < length
                ]
                oracle = [
                    t for t in range(length) if chunk * 64 <= n * (t + 1) - 1 < (chunk + 1) * 64
                ]
                assert source == oracle
                seen.extend(source)
                # Rectangular Q(real64) x K(expanded64) native causal support.
                for qrow in range(min(64, length - block * 64)):
                    cutoff = qrow * n + n - 1 - inner * 64
                    for col in range(64):
                        source_keep = col <= cutoff
                        global_m = chunk * 64 + col
                        oracle_keep = global_m <= n * (block * 64 + qrow + 1) - 1
                        assert source_keep == oracle_keep
                    masks += 1
                mapping_cases += 1
            assert seen == list(range(length))
    assert phase_cases > 0
    # Consequential boundary counterexamples, all in one scalar state.
    record0 = ((F(1),), (F(2),), F(1, 2), F(1, 2))
    record1 = ((F(1),), (F(3),), F(1), F(1, 2))
    s0 = ((F(4),),)
    first = replay_step(s0, record0)
    final = replay_step(first, record1)
    repeated_decay = replay_step(first, (record1[0], record1[1], F(1, 2), record1[3]))
    assert first != final and repeated_decay != final
    assert replay_step(s0, record0) != final  # Accepted1 real token means2 updates.
    result = {
        "passed": True,
        "native_execution": False,
        "recurrence_cases": cases,
        "accepted_prefix_cases": prefix_cases,
        "within_token_chunk_starts": phase_cases,
        "source_readout_chunk_maps": mapping_cases,
        "rectangular_query_masks": masks,
        "oracle": ("Nested token decay/corrections versus independent affine expanded recurrence"),
        "open": [
            "native gate/normalization/rounding",
            "snapshot slot lifetime",
            "approximate split resets are a separate design",
        ],
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
