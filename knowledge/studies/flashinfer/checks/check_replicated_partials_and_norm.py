"""Logical cluster tuple census, phase lifetime and rounded norm checkpoints."""

import itertools
import json
import random
import struct
from pathlib import Path


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def butterfly(values):
    result = [f32(x) for x in values] + [0.0] * (32 - len(values))
    for offset in [16, 8, 4, 2, 1]:
        old = result[:]
        result = [f32(old[lane] + old[lane ^ offset]) for lane in range(32)]
    assert len(set(result)) == 1
    return result[0]


if __name__ == "__main__":
    rng = random.Random(41)
    shape_cases = 0
    deliveries = 0
    for peers in [2, 4, 8]:
        for warps in range(1, 9):
            slots = peers * warps
            if slots > 32:
                continue
            tables = [[None] * slots for _ in range(peers)]
            byte_counts = [0] * peers
            events = list(itertools.product(range(peers), range(peers), range(warps)))
            rng.shuffle(events)
            for destination, sender, warp in events:
                slot = sender * warps + warp
                assert tables[destination][slot] is None
                tables[destination][slot] = [float((slot + 1) * (c + 1)) for c in range(4)]
                byte_counts[destination] += 16
                if byte_counts[destination] < 16 * slots:
                    assert any(x is None for x in tables[destination])
                else:
                    assert all(x is not None for x in tables[destination])
                deliveries += 1
            expected = [sum(range(1, slots + 1)) * (c + 1) for c in range(4)]
            for destination in range(peers):
                assert byte_counts[destination] == 16 * slots
                assert [
                    butterfly([t[c] for t in tables[destination]]) for c in range(4)
                ] == expected
            assert sum(byte_counts) == 16 * peers * slots
            shape_cases += 1

    # A fast peer may start phase1 while a slow peer has not read phase0 yet.
    phase0 = [1.0, 2.0]
    phase1 = [101.0, 102.0]
    slow_bank = phase0[:]
    slow_bank[0] = phase1[0]
    assert sum(slow_bank) != sum(phase0)
    dedicated = [phase0[:], [None, None]]
    dedicated[1][0] = phase1[0]
    assert sum(dedicated[0]) == 3.0

    # Actual FP32 residual square versus algebraically expanded moments.
    u, v = f32(1e8), f32(-1e8 + 8)
    residual = f32(u + v)
    actual_square = f32(residual * residual)
    expanded = f32(f32(f32(u * u) + f32(f32(2 * u) * v)) + f32(v * v))
    assert residual == 8.0 and actual_square == 64.0 and expanded == 0.0

    # The grouped cuTile metadata permutation preserves all distinct coordinates.
    physical = {}
    for row in range(128):
        for col in range(4):
            offset = (row % 32) * 16 + (row // 32) * 4 + col
            assert offset not in physical
            physical[offset] = (row, col)
    assert set(physical) == set(range(512))
    for offset, coordinate in physical.items():
        lane, remainder = divmod(offset, 16)
        quad, col = divmod(remainder, 4)
        assert coordinate == (32 * quad + lane, col)

    # Standalone ceiling MX convention explicitly erases payload when byte==0.
    maximum = 2.0**-120
    raw = f32(maximum * f32(1.0 / 448))
    raw_bits = struct.unpack("<I", struct.pack("<f", raw))[0]
    exponent, mantissa = (raw_bits >> 23) & 255, raw_bits & 0x7FFFFF
    bump = mantissa != 0 and (exponent != 0 or mantissa > 0x400000)
    scale_byte = min(exponent + int(bump), 254)
    assert scale_byte == 0
    stored_scale = 2.0 ** (scale_byte - 127)
    source_multiplier = 0.0
    assert maximum / stored_scale == 128.0 and maximum * source_multiplier == 0.0

    output = {
        "kind": "logical_replicated_cluster_partials_and_norm",
        "status": "passed",
        "native_execution": False,
        "cluster_shape_cases": shape_cases,
        "tuple_deliveries": deliveries,
        "native_binding": {
            "peers": 8,
            "warps": 4,
            "slots": 32,
            "bytes_per_receiver_per_phase": 512,
            "data_bytes_per_cta_two_phases": 1024,
        },
        "phase_bank_alias_witness": "early next-phase send corrupts unread old slot",
        "residual_witness": {
            "residual": residual,
            "square": actual_square,
            "expanded_square": expanded,
        },
        "scale_layout_coordinates": len(physical),
        "standalone_mx_byte0": {
            "maximum": maximum,
            "scale_byte": scale_byte,
            "decoded_ratio": maximum / stored_scale,
            "source_code_input": maximum * source_multiplier,
        },
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(output, indent=2) + "\n")
