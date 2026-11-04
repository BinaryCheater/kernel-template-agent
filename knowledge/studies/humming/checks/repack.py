"Recover one matrix prepack with unique coordinate labels, not repeated INT4 values."

from __future__ import annotations

import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def pack_coordinate_map() -> dict[tuple[int, int], tuple[int, int]]:
    # process.cuh:204,231-265: A=16, B=4, unpreprocessed, no mini-block transpose.
    physical = {}
    for thread in range(32):
        tmp = {}
        for i in range(8):
            for j in range(8):
                for k in range(2):
                    n = i * 8 + thread // 4
                    column = j * 8 + thread % 4 * 2 + k
                    axes = (j // 2, 0, i // 2, i % 2, j % 2, k)
                    flat = 0
                    for axis, extent in zip(axes, (4, 1, 4, 2, 2, 2), strict=True):
                        flat = flat * extent + axis
                    assert flat not in tmp
                    tmp[flat] = (n, column)
        assert len(tmp) == 128
        # process.cuh:8-15,18-54,277-319: both bit fields interleaved (mode=3).
        for row in range(4):
            for word in range(4):
                address = row * 128 + thread * 4 + word
                for slot in range(8):
                    interleaved = slot % 4 * 2 + slot // 4
                    key = (address, slot)
                    assert key not in physical
                    physical[key] = tmp[row * 32 + word * 8 + interleaved]
    return physical


def main() -> None:
    physical = pack_coordinate_map()
    assert len(physical) == 4096
    assert set(physical.values()) == {(n, k) for n in range(64) for k in range(64)}
    # Independent inverse from a physical word/nibble using mixed-radix division.
    for (address, slot), expected in physical.items():
        row, column = divmod(address, 128)
        thread, word = divmod(column, 4)
        flat = row * 32 + word * 8 + slot % 4 * 2 + slot // 4
        axes = []
        for extent in (2, 2, 2, 4, 1, 4):
            flat, axis = divmod(flat, extent)
            axes.append(axis)
        a6, a5, a4, a3, _, a1 = axes
        actual = (16 * a3 + 8 * a4 + thread // 4, (2 * a1 + a5) * 8 + thread % 4 * 2 + a6)
        assert actual == expected
    witness = next(key for key, coord in physical.items() if coord == (17, 43))
    assert witness == (277, 5)
    # An identity permutation is also bijective, but is the wrong source binding.
    address, slot = witness
    row, column = divmod(address, 128)
    thread, word = divmod(column, 4)
    wrong_flat = row * 32 + word * 8 + slot
    correct_flat = row * 32 + word * 8 + slot % 4 * 2 + slot // 4
    assert wrong_flat != correct_flat
    result = {
        "outcome": "PASS",
        "kind": "source_index_replay",
        "gpu_execution": False,
        "binding": {
            "A_bits": 16,
            "B_bits": 4,
            "N": 64,
            "K": 64,
            "preprocess": False,
            "transpose_mini_block": False,
            "packed_k": False,
            "native_dequant": False,
            "interleave_mode": 3,
        },
        "unique_coordinates": 4096,
        "independent_inverse_matches": 4096,
        "witness": {"logical_n": 17, "logical_k": 43, "word": address, "nibble": slot},
        "identity_interleave_mutation_rejected": True,
        "open": [
            "Other consumer layouts/targets beyond composition.py A16/B4 witness",
            "odd-width compression and alternate interleave modes",
            "zero-point preprocessing, expert and padding boundaries",
        ],
    }
    (STUDY / "checks/repack-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: 4096 unique matrix coordinates and inverse; (n=17,k=43) -> "
        "word 277, nibble 5; wrong identity interleave rejected."
    )


if __name__ == "__main__":
    main()
    sys.stdout.write("\n")
