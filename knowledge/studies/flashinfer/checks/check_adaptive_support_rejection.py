"Exact-rational adaptive sampler laws, tie cohorts and remaining-mass witnesses."

import itertools
import json
from fractions import Fraction as F
from functools import cache
from pathlib import Path


def check_law(weights, criterion, limit, refinement):
    n = len(weights)

    def certificate(t):
        return (
            sum(w > t for w in weights)
            if criterion == "above_count"
            else sum((w for w in weights if w > t), F(0))
        )

    target = {i for i, w in enumerate(weights) if w > 0 and certificate(w) < limit}
    target_mass = sum((weights[i] for i in target), F(0))
    assert target and target_mass > 0

    @cache
    def law(low, high):
        support = {i for i, w in enumerate(weights) if w > low}
        assert target <= support and certificate(high) < limit
        mass = sum((weights[i] for i in support), F(0))
        result = [F(0)] * n
        depth = 1
        for i in support:
            probability = weights[i] / mass
            if i in target:
                result[i] += probability
            else:
                new_low, new_high = weights[i], high
                if refinement == "paired_midpoint":
                    midpoint = (weights[i] + high) / 2
                    if certificate(midpoint) < limit:
                        new_high = midpoint
                    else:
                        new_low = midpoint
                child_support = {j for j, w in enumerate(weights) if w > new_low}
                assert target <= child_support < support
                assert new_low < new_high
                child, child_depth = law(new_low, new_high)
                for j in range(n):
                    result[j] += probability * child[j]
                depth = max(depth, 1 + child_depth)
        assert sum(result) == 1
        return tuple(result), depth

    actual, depth = law(F(0), F(1))
    expected = tuple(weights[i] / target_mass if i in target else F(0) for i in range(n))
    assert actual == expected and depth <= n
    return depth


if __name__ == "__main__":
    cases = 0
    max_draws = 0
    for n in range(1, 5):
        for integers in itertools.product(range(4), repeat=n):
            if sum(integers) == 0:
                continue
            weights = tuple(F(w, sum(integers)) for w in integers)
            criteria = [("above_count", k) for k in range(1, n + 1)]
            criteria += [("above_mass", F(j, 4)) for j in range(1, 5)]
            for criterion, limit in criteria:
                for refinement in ["sampled_threshold", "paired_midpoint"]:
                    max_draws = max(max_draws, check_law(weights, criterion, limit, refinement))
                    cases += 1
    weights = [F(4, 10), F(2, 10), F(2, 10), F(2, 10)]
    count_target = [i for i, w in enumerate(weights) if sum(v > w for v in weights) < 2]
    mass_target = [i for i, w in enumerate(weights) if sum(v for v in weights if v > w) < F(1, 2)]
    assert count_target == mass_target == [0, 1, 2, 3]

    # Reject weight0.1 with top-k2; stale q=1 adds an artificial fallback tail.
    correct = [F(2, 3), F(1, 3)]
    stale = [F(3, 5) + F(1, 10) * F(3, 5), F(3, 10) + F(1, 10) * F(2, 5)]
    assert stale == [F(33, 50), F(17, 50)] and stale != correct
    original_above = F(3, 5)
    renormalized_above = original_above / F(9, 10)
    assert original_above < F(13, 20) <= renormalized_above

    report = {
        "kind": "exact_rational_adaptive_support_rejection",
        "status": "passed",
        "native_execution": False,
        "law_cases": cases,
        "maximum_draw_depth": max_draws,
        "domains": "Widths1..4; weights0..3 normalized; two schedules; all k/four mass limits",
        "boundary_tie_cohort": {
            "weights": [float(w) for w in weights],
            "k2_support": count_target,
            "top_p_half_support": mass_target,
        },
        "stale_mass_output_law": {
            "correct": [str(x) for x in correct],
            "stale": [str(x) for x in stale],
        },
        "renormalized_certificate": {
            "original_above": str(original_above),
            "wrong_above": str(renormalized_above),
            "limit": "13/20",
        },
        "boundary": "Exact proposal/certificate arithmetic; native RNG/CDF policy separate",
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
