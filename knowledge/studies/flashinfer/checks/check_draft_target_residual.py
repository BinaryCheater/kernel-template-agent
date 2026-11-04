"Exact draft/target correction laws, conditional chains and diagnostic boundaries."

import itertools
import json
from collections import defaultdict
from fractions import Fraction as F
from pathlib import Path


def correction(draft, target):
    residual = tuple(max(p - q, F(0)) for q, p in zip(draft, target, strict=True))
    return residual, sum(residual, F(0))


def target_row(history):
    first = F(1 + ((len(history) + sum(history)) % 3), 4)
    return first, 1 - first


def draft_row(history):
    first = F(1 + ((2 + len(history) + 2 * sum(history)) % 3), 4)
    return first, 1 - first


if __name__ == "__main__":
    pair_cases = 0
    zero_residual_cases = 0
    for width in range(1, 4):
        distributions = {
            tuple(F(x, sum(xs)) for x in xs)
            for xs in itertools.product(range(3), repeat=width)
            if sum(xs)
        }
        for draft, target in itertools.product(distributions, repeat=2):
            residual, mass = correction(draft, target)
            accepted = tuple(min(q, p) for q, p in zip(draft, target, strict=True))
            reject_mass = 1 - sum(accepted)
            assert reject_mass == mass
            if mass:
                result = tuple(
                    a + reject_mass * r / mass for a, r in zip(accepted, residual, strict=True)
                )
            else:
                assert draft == target
                result = accepted
                zero_residual_cases += 1
            assert result == target
            pair_cases += 1

    # Enumerate an entire two-draft chain and its variable-length output block.
    blocks = defaultdict(F)
    length = 2
    for proposal in itertools.product(range(2), repeat=length):
        history = ()
        path_mass = F(1)
        qs, ps = [], []
        for token in proposal:
            q, p = draft_row(history), target_row(history)
            qs.append(q)
            ps.append(p)
            path_mass *= q[token]
            history += (token,)
        prefix_accept = F(1)
        for r, token in enumerate(proposal):
            acceptance = min(F(1), ps[r][token] / qs[r][token])
            branch_mass = path_mass * prefix_accept * (1 - acceptance)
            if branch_mass:
                residual, mass = correction(qs[r], ps[r])
                assert mass > 0
                for j in range(2):
                    if residual[j] > 0:
                        blocks[(*proposal[:r], j)] += branch_mass * residual[j] / mass
            prefix_accept *= acceptance
        for bonus, p in enumerate(target_row(history)):
            blocks[(*proposal, bonus)] += path_mass * prefix_accept * p
    assert sum(blocks.values()) == 1
    completed = defaultdict(F)

    def continue_target(prefix, probability):
        if len(prefix) == length + 1:
            completed[prefix] += probability
        else:
            for token, p in enumerate(target_row(prefix)):
                continue_target((*prefix, token), probability * p)

    for block, probability in blocks.items():
        continue_target(block, probability)
    for sequence in itertools.product(range(2), repeat=length + 1):
        probability = F(1)
        for t, token in enumerate(sequence):
            probability *= target_row(sequence[:t])[token]
        assert completed[sequence] == probability

    # A diagnostic recheck may pass at the originally rejected position.
    draft = (F(1, 2), F(1, 2))
    target = (F(1, 4), F(3, 4))
    assert F(3, 4) * draft[0] >= target[0]  # Initial rejection.
    diagnostics = int(F(1, 4) * draft[0] < target[0])
    diagnostics += int(F(1, 4) * draft[1] < target[1])
    assert diagnostics == 2  # Emitted prefix is still zero.
    residual, mass = correction(draft, target)
    accepted = [min(q, p) for q, p in zip(draft, target, strict=True)]
    wrong_target_redraw = [a + mass * p for a, p in zip(accepted, target, strict=True)]
    assert wrong_target_redraw == [F(5, 16), F(11, 16)]
    assert tuple(wrong_target_redraw) != target
    # Declared Q must also be the actual draft law.
    actual_draft = [F(3, 4), F(1, 4)]
    wrong_accepted = [
        r * min(F(1), p / q) for r, q, p in zip(actual_draft, draft, target, strict=True)
    ]
    wrong_reject = 1 - sum(wrong_accepted)
    wrong_provenance = [
        a + wrong_reject * r / mass for a, r in zip(wrong_accepted, residual, strict=True)
    ]
    assert wrong_provenance == [F(3, 8), F(5, 8)]

    # Literal source endpoint u1 rejects P==Q, leaving no residual to sample.
    same = (F(1, 2), F(1, 2))
    assert not (F(1) * same[0] < same[0])
    assert correction(same, same)[1] == 0
    report = {
        "kind": "exact_draft_target_residual_correction",
        "status": "passed",
        "native_execution": False,
        "distribution_pair_cases": pair_cases,
        "equal_distribution_zero_residual_cases": zero_residual_cases,
        "conditional_chain": {
            "draft_length": length,
            "completed_target_paths": len(completed),
            "variable_blocks": len(blocks),
            "total_probability": "1",
        },
        "diagnostic_vs_emitted": {"rechecked_accepted": diagnostics, "emitted_drafts": 0},
        "wrong_target_redraw_law": [str(x) for x in wrong_target_redraw],
        "wrong_draft_provenance_law": [str(x) for x in wrong_provenance],
        "source_endpoint_gap": "u1 with equal rows rejects but residual mass is0",
        "source_uniform_draw_counts": {"any_rejection": "L+2", "all_accepted": "L+1"},
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
