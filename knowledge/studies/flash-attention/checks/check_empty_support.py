"""Logical empty-row/frame checks; this does not execute attention on a GPU."""

import json
import math
from pathlib import Path


def normalize(terms, sink, gain):
    if not terms:
        return 0.0, -math.inf if sink is None else sink
    reference = max(x for x, _ in terms)
    denominator = sum(gain * math.exp(x - reference) for x, _ in terms)
    numerator = sum(gain * math.exp(x - reference) * v for x, v in terms)
    if sink is not None:
        denominator += gain * math.exp(sink - reference)
    return numerator / denominator, reference + math.log(denominator) - math.log(gain)


def prefix_step(state, tile, first, gain):
    old_max, old_sum, old_out = state
    maximum = max([x for x, _ in tile] + ([] if first else [old_max]), default=-math.inf)
    safe = 0.0 if maximum == -math.inf else maximum
    correction = 1.0 if first else math.exp(old_max - safe)
    weights = [gain * math.exp(x - safe) for x, _ in tile]
    return (
        maximum,
        old_sum * correction + sum(weights),
        old_out * correction + sum(w * v for w, (_, v) in zip(weights, tile, strict=True)),
    )


if __name__ == "__main__":
    cases = []
    for gain in [1, 256]:
        for sink in [None, math.log(2.0)]:
            for terms in [[], [(math.log(2.0), 3.0), (math.log(4.0), 5.0)]]:
                output, lse = normalize(terms, sink, gain)
                if not terms:
                    assert output == 0
                    assert lse == (-math.inf if sink is None else sink)
                else:
                    expected_denominator = 6.0 + (0.0 if sink is None else 2.0)
                    assert math.isclose(output, 26.0 / expected_denominator)
                    assert math.isclose(lse, math.log(expected_denominator))
                state = (-math.inf, 0.0, 0.0)
                state = prefix_step(state, [(-math.inf, 7.0)], True, gain)
                assert state == (-math.inf, 0.0, 0.0)
                state = prefix_step(state, [(-math.inf, 9.0)], False, gain)
                assert state == (-math.inf, 0.0, 0.0)
                if terms:
                    state = prefix_step(state, terms, False, gain)
                    assert math.isclose(state[2] / state[1], 26.0 / 6.0)
                cases.append(
                    {
                        "gain": gain,
                        "sink": "absent" if sink is None else "finite",
                        "ordinary_terms": len(terms),
                        "result": "passed",
                    }
                )
    # Split0 owns the sink; repeating it in another split changes positive mass.
    owned_once = 26.0 / (2.0 + 4.0 + 2.0)
    duplicated = 26.0 / (2.0 + 4.0 + 2.0 + 2.0)
    assert owned_once != duplicated
    result = {
        "kind": "logical_empty_support_and_common_gain",
        "native_execution": False,
        "cases": cases,
        "duplicate_sink_changes_result": True,
        "remaining": "Native exp/rcp/cast range and precision; sparse/split role synchronization",
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
