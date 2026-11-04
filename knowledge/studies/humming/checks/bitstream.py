"Replay source index arithmetic, not GPU computation, against independent bit identities."

from __future__ import annotations

import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
MASK32 = (1 << 32) - 1


def source_pack(codes: list[int], width: int) -> list[int]:
    """pack_weight.cuh:4-27, transcribed with explicit unsigned truncation."""
    words = [0] * width
    for i, code in enumerate(codes):
        word, offset = divmod(i * width, 32)
        val = code & ((1 << width) - 1)
        words[word] |= (val << offset) & MASK32
        if offset + width > 32:
            words[word + 1] |= val >> (32 - offset)
    return words


def source_unpack(words: list[int], width: int) -> list[int]:
    "pack_weight.cuh:29-50; split extents rather than the oracle's integer extraction."
    codes = []
    for i in range(32):
        word, offset = divmod(i * width, 32)
        first = min(32 - offset, width)
        second = width - first
        val = (words[word] >> offset) & ((1 << first) - 1)
        if second:
            val |= (words[word + 1] & ((1 << second) - 1)) << first
        codes.append(val)
    return codes


def main() -> None:
    evidence = {"kind": "source_arithmetic_replay", "gpu_execution": False, "widths": []}
    for width in range(1, 9):
        cases = [
            [0] * 32,
            [(1 << width) - 1] * 32,
            [(i * 13 + 7) & ((1 << width) - 1) for i in range(32)],
        ]
        for index in range(32):
            for bit in range(width):
                codes = [0] * 32
                codes[index] = 1 << bit
                cases.append(codes)
        crossing = []
        for codes in cases:
            words = source_pack(codes, width)
            stream = sum(code << (i * width) for i, code in enumerate(codes))
            expected = [(stream >> (32 * j)) & MASK32 for j in range(width)]
            assert words == expected
            assert source_unpack(words, width) == codes
        for i in range(32):
            if (i * width) % 32 + width > 32:
                crossing.append(i)
        # The identity oracle must reject a plausible within-word ordering error.
        codes = [0] * 32
        codes[1] = 1
        wrong = source_pack(list(reversed(codes)), width)
        assert wrong != source_pack(codes, width)
        evidence["widths"].append(
            {
                "code_bits": width,
                "cases": len(cases),
                "cross_word_indices": crossing,
                "one_hot_bits": 32 * width,
                "reverse_order_mutation_rejected": True,
            }
        )
    ownership = []
    for length in (32, 64, 992, 1024, 1056, 2016, 2048, 2080):
        visits = []
        for block in range((length + 1023) // 1024):
            for thread in range(32):
                start = block * 1024 + thread * 32
                for width in range(1, 9):
                    out = block * 32 * width + thread * width
                    active = [out + j < length * width // 32 for j in range(width)]
                    assert all(active) == (start < length)
                    assert all(active) or not any(active)
                if start < length:
                    visits.extend(range(start, start + 32))
        assert visits == list(range(length))
        ownership.append({"length": length, "unique_code_visits": len(visits)})
    evidence["ownership"] = ownership
    evidence["outcome"] = "PASS"
    evidence["open"] = [
        "CUDA inactive-thread local initialization and memory safety",
        "compiled kernel behavior and caller allocation behavior",
    ]
    (STUDY / "checks/bitstream-result.json").write_text(json.dumps(evidence, indent=2) + "\n")
    sys.stdout.write(
        "PASS: 1152 one-hot bits across widths 1..8, 24 boundary patterns,"
        " 8 CTA boundary lengths; reversed-code mutation rejected for ever"
        "y width."
    )


if __name__ == "__main__":
    main()
    sys.stdout.write("\n")
