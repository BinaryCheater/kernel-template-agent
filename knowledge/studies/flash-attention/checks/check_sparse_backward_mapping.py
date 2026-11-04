"Logical indexed sums, exponent frames and chunk-coordinate witnesses; no GPU run."

import json
import math
from pathlib import Path

from check_backward_checkpoints import bf16_rne


def scatter(indices, values, rows, initial=None):
    result = [[0.0, 0.0] for _ in range(rows)] if initial is None else [*map(list, initial)]
    for index, value in zip(indices, values, strict=True):
        if 0 <= index < rows:
            result[index] = [a + b for a, b in zip(result[index], value, strict=True)]
    return result


if __name__ == "__main__":
    indices = [2, 0, 2, -1, 3]
    values = [(1, 10), (2, 20), (4, 40), (8, 80), (16, 160)]
    full = scatter(indices, values, 3)
    assert full == [[2.0, 20.0], [0.0, 0.0], [5.0, 50.0]]
    chunked = scatter(indices[:2], values[:2], 3)
    chunked = scatter(indices[2:], values[2:], 3, chunked)
    assert chunked == full
    reset_second = scatter(indices[2:], values[2:], 3)
    assert reset_second != full
    # Assignment, keeping only one duplicate, cannot recover this indexed sum.
    assert list(values[2]) != full[2]

    final_lse2 = math.log2(7)
    references = [1.0, 2.0]
    weights = [[0.5, 1.0], [1.0]]
    decoded = [
        w * math.exp2(reference - final_lse2)
        for reference, tile in zip(references, weights, strict=True)
        for w in tile
    ]
    assert all(math.isclose(x, y) for x, y in zip(decoded, [1 / 7, 2 / 7, 4 / 7], strict=True))
    wrong_frame = [w * math.exp2(2 - final_lse2) for tile in weights for w in tile]
    assert wrong_frame != decoded
    # A narrow shifted checkpoint has a different rounding path from normalized P.
    weight, alpha = 0.502, 0.3
    direct = bf16_rne(weight * alpha)
    two_stage = bf16_rne(bf16_rne(weight) * alpha)
    assert direct != two_stage

    q_length, k_length = 5, 7
    start, end = 1, 3
    k_effective = k_length - q_length + end
    causal_rows = []
    for local_row in range(end - start):
        original_limit = local_row + start + k_length - q_length
        wrong_limit = local_row + k_length - (end - start)
        rebased_limit = local_row + k_effective - (end - start)
        assert rebased_limit == original_limit and wrong_limit != original_limit
        causal_rows.append(
            {"local_row": local_row, "original_limit": original_limit, "wrong_limit": wrong_limit}
        )
    chunks = [(a, min(a + 2, q_length)) for a in range(0, q_length, 2)]
    assert [i for a, b in chunks for i in range(a, b)] == list(range(q_length))
    # Two consumers: keep each row's direct result and reduce through indexed output.
    direct_rows, reduced_rows = [], [[0.0, 0.0] for _ in range(3)]
    for a, b in chunks:
        scratch = values[a:b]
        direct_rows.extend(scratch)
        reduced_rows = scatter(indices[a:b], scratch, 3, reduced_rows)
    assert direct_rows == values and reduced_rows == full

    result = {
        "kind": "logical_sparse_backward_mapping",
        "command": (
            ".venv/bin/python knowledge/studies/flash-attention/checks/check_s"
            "parse_backward_mapping.py"
        ),
        "native_execution": False,
        "indexed_sum": full,
        "duplicate_multiplicity_and_invalid_bounds": "passed",
        "cross_chunk_initialization": "passed",
        "shifted_frames": decoded,
        "wrong_reference_changes_result": True,
        "normalized_bf16": direct,
        "shifted_then_normalized_bf16": two_stage,
        "causal_chunk_rows": causal_rows,
        "complete_chunk_consumers_and_tail": "passed",
        "remaining": (
            "Native atomics and completion, exponent/checkpoint errors, asynch"
            "ronous scratch lifetimes, varlen coordinates and sequence offsets"
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
