"Delta-state transport bounds and active-cut ownership with exact rational oracles."

import itertools
import json
from fractions import Fraction as F
from pathlib import Path

from check_deferred_prefix_state import affine_reference, replay_step


def norm_sq(state):
    return sum((x * x for row in state for x in row), F(0))


def difference(a, b):
    return tuple(tuple(x - y for x, y in zip(r, s, strict=True)) for r, s in zip(a, b, strict=True))


if __name__ == "__main__":
    initial = ((F(1), F(-2)), (F(3), F(1)))
    zero = ((F(0), F(0)), (F(0), F(0)))
    transport_cases = readout_cases = 0
    for length in range(1, 9):
        for recipe in range(5):
            exact, reset = initial, zero
            rho = F(1)
            for step in range(length):
                key = (F((step + recipe) % 3 - 1, 2), F(1, 2))
                beta = F((step + recipe) % 4, 3)
                alpha = F(1, 2) if step % 2 else F(3, 4)
                assert beta * sum((k * k for k in key), F(0)) <= 2
                value = (F(step - recipe, 3), F(recipe, 2))
                record = (key, value, alpha, beta)
                # Independent affine versus prediction/correction implementations.
                exact = affine_reference(exact, record)
                reset = replay_step(reset, record)
                rho *= alpha
                delta = difference(exact, reset)
                assert norm_sq(delta) <= rho * rho * norm_sq(initial)
                transport_cases += 1
                for query in [(F(1), F(0)), (F(1, 3), F(-1, 2))]:
                    projected_sq = sum(
                        (
                            sum((x * q for x, q in zip(row, query, strict=True)), F(0)) ** 2
                            for row in delta
                        ),
                        F(0),
                    )
                    query_sq = sum((q * q for q in query), F(0))
                    assert projected_sq <= query_sq * rho * rho * norm_sq(initial)
                    readout_cases += 1
    # Gate-product-only counterexamples: huge seed and noncontractive correction.
    tiny_gate = F(1, 2**24)
    large = ((F(2**30), F(0)), (F(0), F(0)))
    no_insert = ((F(0), F(0)), (F(0), F(0)), tiny_gate, F(0))
    large_error = norm_sq(replay_step(large, no_insert))
    assert large_error == 64**2
    huge_key = ((F(2**13), F(0)), (F(0), F(0)), tiny_gate, F(1))
    unit = ((F(1), F(0)), (F(0), F(0)))
    amplified = norm_sq(replay_step(unit, huge_key))
    assert amplified > 1  # Gate2^-24 alone passed, but delta expanded by almost4.
    partitions = owned_chunks = aligned_leading = 0
    for planned_count in range(1, 9):
        cuts = [6 * j for j in range(planned_count)]
        end = cuts[-1] + 10
        for choices in itertools.product([False, True], repeat=planned_count - 1):
            active = [True, *choices]
            oracle = [cut for cut, live in zip(cuts, active, strict=True) if live]
            expected_ranges = list(zip(oracle, [*oracle[1:], end], strict=True))
            ranges = []
            owners = []
            final_owners = 0
            for i, cut in enumerate(cuts):
                if not active[i]:
                    continue  # Rejected cut produces no writes.
                stop = next((cuts[j] for j in range(i + 1, len(cuts)) if active[j]), end)
                warmup = min(cut, 1 + i % 8) if cut else 0
                start = cut - warmup
                if (stop - start) % 2:
                    start -= 1
                    aligned_leading += 1
                assert 0 <= start <= cut < stop
                ranges.append((cut, stop))
                owners.extend(range(cut, stop))
                assert not set(range(start, cut)) & set(range(cut, stop))
                final_owners += stop == end
            assert ranges == expected_ranges
            assert owners == list(range(end)) and final_owners == 1
            partitions += 1
            owned_chunks += len(owners)
    result = {
        "passed": True,
        "native_execution": False,
        "state_transport_prefixes": transport_cases,
        "readout_bounds": readout_cases,
        "active_cut_partitions": partitions,
        "owned_chunk_occurrences": owned_chunks,
        "pair_alignment_extra_leading_cases": aligned_leading,
        "counterexamples": [
            "gate2^-24 with seed2^30 leaves error64",
            "gate2^-24 with key norm2^13 amplifies unit-state error",
        ],
        "oracle": "Independent affine/reset recurrence and explicit active interval list",
        "open": [
            "native approximate-log/exp upper bounds",
            "native rounding budgets",
            "input seed/key/readout bounds",
            "GPU seed-reader publication/liveness",
        ],
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
