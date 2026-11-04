"Logical value compensation and mass-role witnesses; no native GPU arithmetic."

import json
import math
from pathlib import Path


def grouped_mean_restore(tiles, group_size, *, rescale_pending=True, flush=True):
    reference, numerator, mass = 0.0, 0.0, 0.0
    pending = []
    for index, (center, residuals, logits) in enumerate(tiles):
        new_reference = max(reference, max(logits))
        correction = 2 ** (reference - new_reference)
        numerator *= correction
        mass *= correction
        if rescale_pending:
            pending = [(u * correction, c) for u, c in pending]
        weights = [2 ** (x - new_reference) for x in logits]
        u = sum(weights)
        numerator += sum(w * r for w, r in zip(weights, residuals, strict=True))
        mass += u
        pending.append((u, center))
        reference = new_reference
        if (index + 1) % group_size == 0:
            numerator += sum(u * c for u, c in pending)
            pending = []
    if flush:
        numerator += sum(u * c for u, c in pending)
    return numerator / mass


if __name__ == "__main__":
    grouped_cases = []
    for count in (1, 15, 16, 17, 33):
        tiles = []
        for t in range(count):
            # A physical128-key tile with two valid keys and a one-key final tail.
            logits = [float(t // 3), float(t // 3 - 1)]
            residuals = [0.25, -0.5]
            if t == count - 1:
                logits, residuals = logits[:1], residuals[:1]
            padding = 128 - len(logits)
            logits += [-math.inf] * padding
            residuals += [0.0] * padding
            tiles.append((float(2 + t % 4), residuals, logits))
        reference = max(x for _, _, xs in tiles for x in xs)
        denominator = sum(2 ** (x - reference) for _, _, xs in tiles for x in xs)
        expected = (
            sum(
                2 ** (x - reference) * (center + residual)
                for center, rs, xs in tiles
                for residual, x in zip(rs, xs, strict=True)
            )
            / denominator
        )
        assert math.isclose(grouped_mean_restore(tiles, 1), expected, rel_tol=1e-14)
        assert math.isclose(grouped_mean_restore(tiles, 16), expected, rel_tol=1e-14)
        if count > 1:
            assert not math.isclose(
                grouped_mean_restore(tiles, 16, rescale_pending=False), expected
            )
        if count % 16:
            assert grouped_mean_restore(tiles, 16, flush=False) != expected
        grouped_cases.append({"tiles": count, "expected": expected})
    # The center multiplies rounded-P mass, just as residual PV does.
    rounded_weights = [1.0, 0.3125]
    original_weights = [1.0, 0.30]
    center, residuals = 8.0, [0.5, -0.25]
    residual_pv = sum(w * r for w, r in zip(rounded_weights, residuals, strict=True))
    restored = residual_pv + center * sum(rounded_weights)
    wrong_mean_mass = residual_pv + center * sum(original_weights)
    assert restored != wrong_mean_mass
    # Simulated narrow value codebooks isolate residual generation/mass roles.
    values = [1.1, 2.2, 3.3]
    base = [round(v * 4) / 4 for v in values]
    errors = [v - b for v, b in zip(values, base, strict=True)]
    selected = sorted(range(len(values)), key=lambda j: abs(errors[j]), reverse=True)[:2]
    repairs = {j: round(errors[j] * 32) / 32 for j in selected}
    weights = [1.0, 0.5, 0.25]
    mass = sum(weights)
    base_o = sum(w * b for w, b in zip(weights, base, strict=True))
    repair_o = sum(weights[j] * r for j, r in repairs.items())
    result = (base_o + repair_o) / mass
    expected = sum(weights[j] * (base[j] + repairs.get(j, 0)) for j in range(3)) / mass
    assert math.isclose(result, expected)
    assert (base_o + repair_o) / (mass + sum(weights[j] for j in selected)) != result
    assert (base_o + 2 * repair_o) / mass != result
    assert (base_o + 0.5 * repair_o) / mass != result
    # Replacing a base generation without recomputing its residual changes Vhat.
    changed_base = base.copy()
    changed_base[selected[0]] += 0.25
    assert (
        sum(weights[j] * (changed_base[j] + repairs.get(j, 0)) for j in range(3)) / mass != result
    )
    correction = 2**-4
    assert math.isclose((base_o * correction + repair_o * correction) / (mass * correction), result)
    assert (base_o * correction + repair_o) / (mass * correction) != result
    # 130 originals: full prefix, one128-row repair tile, moved two-row tail.
    original_to_physical = [j if j < 128 else j + 128 for j in range(130)]
    original_rows = set(original_to_physical)
    repair_rows = set(range(128, 256))
    assert original_rows.isdisjoint(repair_rows)
    assert len(original_rows) == 130 and {256, 257}.issubset(original_rows)
    naive_original_rows = set(range(130))
    assert naive_original_rows != original_rows
    assert len(naive_original_rows & repair_rows) == 2
    output = {
        "kind": "logical_value_compensation_check",
        "native_execution": False,
        "status": "passed",
        "mean_group_cases": grouped_cases,
        "residual_replay_output": result,
        "checks": [
            "mean_identity",
            "pending_reference_updates",
            "final_group_flush",
            "rounded_probability_mass",
            "selected_residual_identity",
            "duplicate_mass",
            "duplicate_selection",
            "replay_weight_mismatch",
            "stale_base_generation",
            "replay_frame",
            "relocated_tail_roles",
        ],
        "numeric_boundary": (
            "Dyadic weights and simulated narrow value codebooks isolate invar"
            "iants; no native ExpCast/BF16/UMMA equivalence is asserted."
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(output, indent=2) + "\n")
