"Discriminating scalar replays of scale encoding, packing and SM100 placement."

from __future__ import annotations

import itertools
import json
import math
import struct
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def fp32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def source_pack(bits: list[int]) -> int:
    assert len(bits) == 4
    assert all(value & 0x807FFFFF == 0 for value in bits)
    return (bits[0] >> 23 | bits[1] >> 15 | bits[2] >> 7 | bits[3] << 1) & 0xFFFFFFFF


def decode(code: int) -> float:
    return math.nan if code == 255 else 2.0 ** (code - 127)


def grouped_pack(rows: list[int], lengths: tuple[int, ...]) -> list[int]:
    "Transcribe source packed-row owner search and cumulative padding correction."
    result = []
    for packed_idx in range(sum((length + 3) // 4 for length in lengths)):
        prefix_input = prefix_packed = padding = 0
        for length in lengths:
            start, end = prefix_input, prefix_input + length
            prefix_input = end
            prefix_packed += (length + 3) // 4
            if packed_idx < prefix_packed:
                break
            if length % 4:
                padding += 4 - length % 4
        bits = []
        for j in range(4):
            idx = packed_idx * 4 + j - padding
            bits.append(rows[idx] if start <= idx < end else 0)
        result.append(source_pack(bits))
    return result


def main() -> None:
    source = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(source["tree"])
    pack_body = (root / "deep_gemm/include/deep_gemm/impls/smxx_layout.cuh").read_text()
    for marker in (
        "values[0] >> 23u",
        "values[1] >> 15u",
        "values[2] >>  7u",
        "values[3] <<  1u",
        "num_padding_sf_rows += 4 - remainder",
        "sf_row_idx >= group_sf_row_start and sf_row_idx < group_sf_row_end",
    ):
        assert marker in pack_body
    modules = json.loads((STUDY / "inputs/submodule-sources.json").read_text())
    cutlass = Path(
        next(item["tree"] for item in modules if item["superproject_path"].endswith("cutlass"))
    )
    decoder_body = (cutlass / "include/cutlass/float8.h").read_text()
    assert "x.storage == 0x00" in decoder_body and "0x00400000" in decoder_body
    assert "x.storage == 0xFF" in decoder_body
    for code in range(1, 255):
        assert fp32(code << 23) == decode(code)
        assert source_pack([code << 23, 0, 0, 0]) & 255 == code
    # Guard-only acceptance is insufficient for an exact live-scale domain.
    assert source_pack([0, 0, 0, 0]) == 0
    assert decode(0) != fp32(0)
    assert math.isinf(fp32(255 << 23)) and math.isnan(decode(255))
    assert source_pack([255 << 23, 0, 0, 0]) & 255 == 255
    # Distinct bytes, rows, K slices and batches prevent permutation errors from hiding.
    mn, sf_k, groups = 6, 5, 2
    pitch, packed_k = 8, 2
    output: dict[int, int] = {}
    logical: dict[tuple[int, int, int], int] = {}
    for group, row, pack in itertools.product(range(groups), range(mn), range(packed_k)):
        codes = [
            (group * 53 + row * 7 + pack * 4 + j + 1) if pack * 4 + j < sf_k else 0
            for j in range(4)
        ]
        word = source_pack([code << 23 for code in codes])
        address = group * pitch * packed_k + pack * pitch + row
        assert address not in output
        output[address] = word
        for j in range(4):
            logical[group, row, pack * 4 + j] = codes[j]
    for (group, row, k), expected in logical.items():
        address = group * pitch * packed_k + (k // 4) * pitch + row
        assert (output[address] >> ((k % 4) * 8)) & 255 == expected
    assert output[pitch] != output[1]  # K-slice and MN coordinate swaps are distinguishable.
    grouped_cases = 0
    for lengths in itertools.product((1, 2, 3, 4, 5, 8), repeat=3):
        codes = list(range(1, sum(lengths) + 1))
        words, start = [], 0
        for length in lengths:
            group_codes = codes[start : start + length]
            for offset in range(0, length, 4):
                chunk = group_codes[offset : offset + 4]
                words.append(sum(code << (8 * j) for j, code in enumerate(chunk)))
            start += length
        assert grouped_pack([code << 23 for code in codes], lengths) == words
        grouped_cases += 1
    assert grouped_pack([code << 23 for code in range(1, 9)], (1, 3, 4)) != [0x04030201, 0x08070605]
    # UTCCP pre-transpose: source word (i,lane) moves from i*32+lane to lane*4+i.
    tags = [(row, byte) for row in range(128) for byte in range(4)]
    words = [tags[4 * row : 4 * row + 4] for row in range(128)]
    transposed = [None] * 128
    for lane in range(32):
        for i in range(4):
            transposed[lane * 4 + i] = words[i * 32 + lane]
    assert all(transposed[(row % 32) * 4 + row // 32] == words[row] for row in range(128))
    assert transposed[1] != words[1]
    result = {
        "outcome": "PASS",
        "exact_scale_exponents": 254,
        "source_guard_counterexamples": {
            "zero": "UE8M0 code0 decodes to2^-127",
            "positive_infinity": "UE8M0 code255 is NaN",
        },
        "mn_major_witness": {
            "mn": mn,
            "sf_k": sf_k,
            "groups": groups,
            "pitch": pitch,
            "unique_written_words": len(output),
        },
        "grouped_packing_domains": grouped_cases,
        "utccp_tagged_words": 128,
        "mutations_rejected": ["flat_across_group_packing", "MN_K_swap", "no_UTCCP_transpose"],
        "boundary": "Scalar bit/index replay with retained decoder; no GPU or UTCCP execution.",
    }
    (STUDY / "checks/scale-storage-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        f"PASS: 254 exact exponents; {grouped_cases} grouped packings; 128 UTCCP tags.\n"
    )


if __name__ == "__main__":
    main()
