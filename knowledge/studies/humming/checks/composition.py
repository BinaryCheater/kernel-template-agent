"Compose source pack/g2s/s2r/dequant indices with the documented FP16 MMA B layout."

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

from repack import pack_coordinate_map

STUDY = Path(__file__).resolve().parents[1]


def half(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def main() -> None:
    packed = pack_coordinate_map()
    visited = []
    witness = None
    for k_warp in range(2):
        for n_warp in range(4):
            for iteration in range(2):
                for lane in range(32):
                    # loader_b.cuh:119-140, uint2 load, two K warps and N=32 half groups.
                    idx = 32 * (n_warp // 2) + lane + 128 * k_warp
                    base_word = iteration * 256 + (idx * 2 + n_warp % 2) * 2
                    for fragment in range(2):
                        word = base_word + fragment
                        row, column = divmod(word, 256)
                        n_block, local_column = divmod(column, 128)
                        for pair in range(4):
                            for element in range(2):
                                # dequant_b1248: q nibble pair (i,i+4), no FP reversal.
                                actual_n, actual_k = packed[
                                    (row * 128 + local_column, pair + 4 * element)
                                ]
                                actual_n += 64 * n_block
                                # PTX mma.m16n8k16 B.col: two half2 registers per N=8.
                                expected_n = n_warp * 32 + fragment * 16 + pair // 2 * 8 + lane // 4
                                expected_k = k_warp * 32 + iteration * 16 + pair % 2 * 8
                                expected_k += lane % 4 * 2 + element
                                assert (actual_n, actual_k) == (expected_n, expected_k)
                                visited.append((actual_n, actual_k))
                                if (actual_n, actual_k) == (17, 43):
                                    witness = {
                                        "n_warp": n_warp,
                                        "k_warp": k_warp,
                                        "iteration": iteration,
                                        "lane": lane,
                                        "n_fragment": fragment,
                                        "pair": pair,
                                        "element": element,
                                        "global_word": word,
                                    }
    assert len(visited) == len(set(visited)) == 8192
    assert set(visited) == {(n, k) for n in range(128) for k in range(64)}
    # dequant_single.cuh:34-44: exact half mantissa construction and subtraction.
    for zero in (5, 8, 13):
        for code in range(16):
            assert half(0x6400 | code) - half(0x6400 + zero) == code - zero
    # Unequal groups falsify moving an arbitrary group scale across the K reduction.
    correct = sum([2] * 32 + [3] * 32)
    wrong = sum([1] * 64) * 2
    assert correct == 160 and wrong == 128
    result = {
        "outcome": "PASS",
        "kind": "source_layout_and_numeric_replay",
        "gpu_execution": False,
        "binding": {
            "N": 128,
            "K": 64,
            "A_bits": 16,
            "B_bits": 4,
            "block": [16, 128, 64],
            "warp": [16, 32, 32],
        },
        "B_fragment_coordinates": len(visited),
        "witness_17_43": witness,
        "exact_half_decode_cases": 48,
        "unequal_group_scales": {"correct": correct, "invalid_postscale": wrong},
        "premises": [
            "legacy_load_2d preserves the packed row's word order",
            "mma.sync m16n8k16 row.col FP16 B fragment uses PTX ISA 9.4 layout",
        ],
        "open": [
            "A fragment and accumulator output composition",
            "compiled CUDA execution",
            "other widths, targets and native/packed-K layouts",
        ],
    }
    (STUDY / "checks/composition-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: 8192 B fragment coordinates; 48 exact half decodes; unequal"
        "-group scale mutation rejected.\n"
    )


if __name__ == "__main__":
    main()
