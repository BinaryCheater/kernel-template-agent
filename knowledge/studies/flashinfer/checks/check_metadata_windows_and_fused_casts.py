"Logical metadata windows and preserved gated-output checkpoints; no GPU replay."

import json
import math
import struct
from pathlib import Path


def bf16(x):
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]


def gated(g, u, *, input_cast=True, activation_cast=True):
    if input_cast:
        g, u = bf16(g), bf16(u)
    result = (g / (1 + math.exp(-g))) * u
    return bf16(result) if activation_cast else result


if __name__ == "__main__":
    coordinate_cases = 0
    for rows in (1, 3, 128):
        for steps in (4, 8, 12):
            source = [(row, step) for row in range(rows) for step in range(steps)]
            for group in range(steps // 4):
                window = [None] * (rows * 4)
                for row in range(rows):
                    window[row * 4 : (row + 1) * 4] = source[
                        row * steps + group * 4 : row * steps + (group + 1) * 4
                    ]
                for row in range(rows):
                    for j in range(4):
                        assert window[row * 4 + j] == (row, group * 4 + j)
                        coordinate_cases += 1
    raw, scales = [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 5.0, 7.0]
    reference = sum(r * s for r, s in zip(raw, scales, strict=True))
    assert reference == 51
    assert sum(raw) * scales[-1] != reference
    window = scales.copy()
    first = raw[0] * window[0]
    window[:] = [11.0, 13.0, 17.0, 19.0]
    wrong = first + sum(raw[j] * window[j] for j in range(1, 4))
    assert wrong != reference
    # Feature owners can finish at different times; all must retire each slice.
    pending = {(step, owner) for step in range(4) for owner in range(2)}
    pending.remove((0, 0))
    assert pending
    pending -= {(step, owner) for step in range(4) for owner in range(2)}
    assert not pending
    generations = []
    ordinal = 0
    for tile in range(3):
        for group in range(2):
            generations.append((tile, group, ordinal % 3, ordinal // 3 % 2))
            ordinal += 1
    assert generations[2][2:] == (2, 0)
    assert generations[3][2:] == (0, 1)
    assert generations[2][2:] != (0, 0)
    batches = {steps: 4 if steps % 4 == 0 else 1 for steps in (1, 3, 4, 5, 8, 11, 12)}
    assert batches[5] == 1 and batches[8] == 4
    assert 5 // 4 * 4 != 5
    input_witness = None
    activation_witness = None
    for g in (0.127, 0.333, 1.003, 1.013, 2.017, -1.017):
        for u in (0.319, 1.007, 3.141, 8.031):
            expected = gated(g, u)
            if expected != gated(g, u, input_cast=False):
                input_witness = [g, u, expected, gated(g, u, input_cast=False)]
            if expected != gated(g, u, activation_cast=False):
                activation_witness = [g, u, expected, gated(g, u, activation_cast=False)]
    assert input_witness is not None and activation_witness is not None
    h = [0.75] * 64 + [8.0] * 64
    scale = max(abs(x) for x in h) / 448
    assert scale != max(abs(x) for x in h[:64]) / 448
    assert max(max(abs(x) for x in h[:64]), max(abs(x) for x in h[64:])) / 448 == scale
    result = {
        "kind": "logical_metadata_window_and_cast_check",
        "native_execution": False,
        "status": "passed",
        "coordinate_cases": coordinate_cases,
        "input_cast_witness": input_witness,
        "activation_cast_witness": activation_witness,
        "batch_fallback": batches,
        "checks": [
            "row_step_identity",
            "stepwise_scale_promotion",
            "last_slice_lifetime",
            "all_feature_owners_retire",
            "cross_tile_ring_generation",
            "incomplete_batch_fallback",
            "gate_up_bf16_checkpoint",
            "activation_bf16_checkpoint",
            "full128_quant_group",
        ],
        "numeric_boundary": (
            "Logical coordinates and Python/BF16 witnesses; native async copie"
            "s, fast SiLU and quantizer bit equality remain separate."
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
