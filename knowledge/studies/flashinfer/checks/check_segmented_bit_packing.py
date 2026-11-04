"""Boolean segment origins, byte order and zero-tail round trips."""

import itertools
import json
from pathlib import Path


def pack(segments, order):
    output, prefix = [], [0]
    for segment in segments:
        for start in range(0, len(segment), 8):
            byte = 0
            for t in range(8):
                value = segment[start + t] if start + t < len(segment) else False
                byte |= int(value) << (7 - t if order == "big" else t)
            output.append(byte)
        prefix.append(len(output))
    return output, prefix


def verify(segments, order):
    output, prefix = pack(segments, order)
    assert prefix[0] == 0
    assert prefix[-1] == sum((len(s) + 7) // 8 for s in segments)
    coordinates = set()
    for s, segment in enumerate(segments):
        assert prefix[s + 1] - prefix[s] == (len(segment) + 7) // 8
        for i, value in enumerate(segment):
            coordinate = prefix[s] + i // 8, 7 - i % 8 if order == "big" else i % 8
            assert coordinate not in coordinates
            coordinates.add(coordinate)
            byte, bit = coordinate
            assert bool((output[byte] >> bit) & 1) == value
        for i in range(len(segment), 8 * (prefix[s + 1] - prefix[s])):
            byte, bit = prefix[s] + i // 8, 7 - i % 8 if order == "big" else i % 8
            assert (output[byte] >> bit) & 1 == 0
    return output, prefix


if __name__ == "__main__":
    cases = 0
    for order in ["big", "little"]:
        for length in range(9):
            for pattern in itertools.product([False, True], repeat=length):
                verify([list(pattern)], order)
                cases += 1
        for lengths in itertools.product([0, 1, 7, 8, 9, 15, 16, 17], repeat=3):
            segments = [
                [((s * 3 + i * 5) % 7) < 3 for i in range(length)]
                for s, length in enumerate(lengths)
            ]
            verify(segments, order)
            cases += 1
        verify([[i % 5 < 2 for i in range(n)] for n in [2047, 2048, 2049]], order)
        cases += 1
    assert cases == 2048
    segments = [[True, False, True], [False, True, True, False, True]]
    output, prefix = verify(segments, "big")
    flat, _ = pack([list(itertools.chain.from_iterable(segments))], "big")
    assert output == [0xA0, 0x68] and flat == [0xAD]
    assert prefix == [0, 1, 2]
    assert pack([[True]], "big")[0] == [128]
    assert pack([[True]], "little")[0] == [1]
    assert pack([[], [], []], "big") == ([], [0, 0, 0, 0])
    report = {
        "kind": "logical_segmented_bit_packing",
        "status": "passed",
        "native_execution": False,
        "roundtrip_cases": cases,
        "byte_patterns_and_tails": "exhaustive lengths0..8 for both orders",
        "multi_segment_lengths": [0, 1, 7, 8, 9, 15, 16, 17],
        "cta_block_boundaries": [2047, 2048, 2049],
        "separate_vs_flat": {"separate_bytes": output, "prefix": prefix, "flat_bytes": flat},
        "empty_segments": "zero bytes and repeated prefix entries",
        "native_boundary": "no CUB, CUDA or giant-extent execution claimed",
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
