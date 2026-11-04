"""Separating numerical-policy and metadata-coordinate witnesses."""

import json
import struct
from pathlib import Path


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def unbits(x):
    return struct.unpack("<f", struct.pack("<I", x))[0]


def source_scale(a):
    e0 = ((bits(a) >> 23) & 255) - 135
    threshold = unbits(((e0 + 135) << 23) | 0x600000)
    e = max(e0 + int(a > threshold), -127)
    return e


cases = []
for e in [-100, -1, 0, 10, 100]:
    a = f32(448 * (2.0**e))
    for value in [unbits(bits(a) - 1), a, unbits(bits(a) + 1)]:
        selected = source_scale(value)
        # Independent exact binary comparisons avoid log2 near a rounding boundary.
        expect = -127
        while 448 * (2.0**expect) < value:
            expect += 1
        assert selected == expect
        cases.append({"amax": value, "exponent": selected})
assert source_scale(0.0) == -127
rows, cols = 129, 7
pr, pc = 256, 8
addresses = set()
for r in range(rows):
    for c in range(cols):
        src = (
            ((r // 128) * ((cols + 3) // 4) + c // 4) * 512
            + (r % 32) * 16
            + ((r % 128) // 32) * 4
            + c % 4
        )
        rule = (r // 128) * pc * 128 + (c // 4) * 512 + (r % 32) * 16 + ((r // 32) % 4) * 4 + c % 4
        assert src == rule and src not in addresses and src < pr * pc
        addresses.add(src)
assert ((33 % 32) * 16 + ((33 % 128) // 32) * 4 + 1) == 21
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "outcome": "passed",
            "ceiling_boundary_cases": cases,
            "zero_exponent": -127,
            "layout_valid_coordinates": len(addresses),
            "layout_extent": [rows, cols],
            "scope": (
                "Host FP32 bit/coordinate witness; no native payload conversion or MMA mapping"
            ),
        },
        indent=2,
    )
    + "\n"
)
