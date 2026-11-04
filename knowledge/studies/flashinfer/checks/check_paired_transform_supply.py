"Logical pair supply, in-place ordering, checkpoint and cache-coordinate witnesses."

import json
import math
import struct
from pathlib import Path


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]


def rotate(a, b, c, s):
    return f32(f32(a * c) - f32(b * s)), f32(f32(b * c) + f32(a * s))


def pair_map(width, active, pairing):
    pairs = [
        (2 * j, 2 * j + 1) if pairing == "adjacent" else (j, j + active // 2)
        for j in range(active // 2)
    ]
    assert sorted(i for pair in pairs for i in pair) == list(range(active))
    assert all(a < width and b < width for a, b in pairs)
    return pairs


if __name__ == "__main__":
    cases = 0
    for width in [8, 16, 32, 64, 128]:
        for active in range(2, width + 1, 2):
            for pairing in ["adjacent", "split_halves"]:
                values = [f32(i + 0.125) for i in range(width)]
                expected = values[:]
                partners = {}
                for j, (a, b) in enumerate(pair_map(width, active, pairing)):
                    c, s = f32(0.3 + j / 1000), f32(0.7 - j / 1000)
                    expected[a], expected[b] = rotate(values[a], values[b], c, s)
                    partners[a], partners[b] = (b, -1, c, s), (a, 1, c, s)
                duplicate = values[:]
                for owner, (partner, sign, c, s) in partners.items():
                    duplicate[owner] = f32(f32(values[owner] * c) + sign * f32(values[partner] * s))
                assert duplicate == expected
                assert expected[active:] == values[active:]
                cases += 1
    assert cases == 248
    exchange_coordinates = 0
    for width in [64, 128, 256]:
        per_lane = width // 32
        for lane in range(32):
            for slot in range(per_lane):
                feature = lane * per_lane + slot
                exchanged = (lane ^ 16) * per_lane + slot
                assert exchanged == (feature + width // 2) % width
                exchange_coordinates += 1

    old = [1.0, 2.0]
    expected = rotate(*old, 0.0, 1.0)
    broken = old[:]
    broken[0] = rotate(*broken, 0.0, 1.0)[0]
    broken[1] = rotate(*broken, 0.0, 1.0)[1]
    assert expected == (-2.0, 1.0) and tuple(broken) == (-2.0, -2.0)

    levels = [i * 2.0**-9 for i in range(8)]
    levels += [
        (1 + m / 8) * 2.0**e for e in range(-6, 9) for m in range(8) if (1 + m / 8) * 2.0**e <= 448
    ]

    def fp8(value):
        i = min(range(len(levels)), key=lambda i: (abs(levels[i] - abs(value)), i % 2))
        return math.copysign(levels[i], value)

    witness = None
    for i in range(1, 4097):
        x = f32(1 + i / 4096)
        y = f32(0.125)
        with_cast = rotate(bf16(x), bf16(y), f32(0.6), f32(0.8))[0]
        without_cast = rotate(x, y, f32(0.6), f32(0.8))[0]
        if fp8(with_cast) != fp8(without_cast):
            witness = {
                "x": x,
                "y": y,
                "with_bf16_checkpoint": fp8(with_cast),
                "without_bf16_checkpoint": fp8(without_cast),
            }
            break
    assert witness is not None

    heads = [[1.0, 1.0], [3.0, 3.0]]
    across = [[x / math.sqrt(5) for x in head] for head in heads]
    per_head = [[x / math.sqrt(sum(v * v for v in head) / 2) for x in head] for head in heads]
    assert per_head == [[1.0, 1.0], [1.0, 1.0]] and across != per_head
    page_size, indptr, pages = 4, [0, 2, 3], [9, 4, 17]
    positions, batches, pos_ids = [4, 0], [0, 1], [11, 7]
    addresses = []
    for batch, position in zip(batches, positions, strict=True):
        page_iter, entry = divmod(indptr[batch] * page_size + position, page_size)
        addresses.append((pages[page_iter], entry))
    assert addresses == [(4, 0), (17, 0)]
    assert pos_ids != positions
    report = {
        "kind": "logical_paired_transform_supply",
        "status": "passed",
        "native_execution": False,
        "pair_and_suffix_cases": cases,
        "xor16_coordinates": exchange_coordinates,
        "inplace_witness": {"correct": expected, "early_store": broken},
        "fp8_does_not_erase_checkpoint_difference": witness,
        "norm_domain_witness": {"across_heads": across, "per_head": per_head},
        "cache_coordinate_witness": {
            "rope_positions": pos_ids,
            "cache_positions": positions,
            "addresses": addresses,
        },
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
