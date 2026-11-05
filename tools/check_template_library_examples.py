"""Check local arithmetic and boundary examples published in the curated guides."""

from __future__ import annotations

import math
from pathlib import Path

import yaml


def check_examples() -> list[str]:
    curated = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "knowledge/library/curation.yaml").read_text()
    )
    required = {
        "output_tiling@2",
        "reduction_partition@1",
        "compute_instruction_selection@2",
        "multi_buffer_rotation@2",
        "masked_tile_pruning@2",
        "online_rescaled_reduction@2",
    }
    assert required <= curated.keys()
    # Ownership includes exactly the valid output coordinates, including residue tiles.
    coordinates = [
        (bi * 32 + i, bj * 64 + j)
        for bi in range(4)
        for bj in range(16)
        for i in range(32)
        for j in range(64)
        if bi * 32 + i < 127 and bj * 64 + j < 1000
    ]
    assert len(coordinates) == len(set(coordinates)) == 127 * 1000
    reduction = [g * 32 + k for g in range(25) for k in range(32) if g * 32 + k < 769]
    assert reduction == list(range(769))
    assert (32 // 16) * (64 // 8) * (32 // 16) == 32
    assert 32 * (16 * 8 * 16) == 32 * 64 * 32
    assert 1 <= 2 - 2 + 1 + 0 - 1 * 0 + 0
    assert not 1 <= 2 - 2 + 0 + 0 - 0 * 0 + 0
    assert 2 * 4 * (32 * 32 + 32 * 64) == 24576
    # Exhaust the documented mask derivation over every tile of the odd-length example.
    length, bm, bn = 513, 32, 64
    for q0 in range(0, length, bm):
        q1 = min(q0 + bm, length)
        end = math.ceil(q1 / bn)
        core = (q0 + 1) // bn
        assert 0 <= core <= end <= math.ceil(length / bn)
        for key_tile in range(math.ceil(length / bn)):
            for i in range(q0, q1):
                for j in range(key_tile * bn, (key_tile + 1) * bn):
                    valid = j < length and j <= i
                    if key_tile >= end:
                        assert not valid
                    if key_tile < core:
                        assert valid
    assert (math.ceil(96 / 64), (64 + 1) // 64) == (2, 1)
    assert (math.ceil(513 / 64), (512 + 1) // 64) == (9, 8)
    for offset in [0.0, -1000.0]:
        r, denominator, numerator = None, 0.0, 0.0
        # None is a fully masked prefix tile, which must not activate a reference.
        for score, value in [(None, 99.0), (offset, 2.0), (offset + math.log(2.0), 4.0)]:
            if score is None:
                assert r is None and denominator == numerator == 0.0
                continue
            if r is None:
                r, denominator, numerator = score, 1.0, value
                continue
            new_r = max(r, score)
            correction = math.exp(r - new_r)
            weight = math.exp(score - new_r)
            denominator = correction * denominator + weight
            numerator = correction * numerator + weight * value
            r = new_r
        assert math.isclose(denominator, 1.5)
        assert math.isclose(numerator, 5.0)
        assert math.isclose(numerator / denominator, 10 / 3)
    assert math.exp(-1000.0) == 0.0  # Demonstrates why an arbitrary zero reference fails.
    return [
        "127x1000 output ownership",
        "K=769 tail partition",
        "instruction product count",
        "buffer lifetime accepted/rejected bindings",
        "fixed-model shared storage bytes",
        "all causal tile boundaries for S=513",
        "online common-frame scalar recurrence and negative-offset initialization",
    ]


if __name__ == "__main__":
    for result in check_examples():
        print(f"PASS {result}")
