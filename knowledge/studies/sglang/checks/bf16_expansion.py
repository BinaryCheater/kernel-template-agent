"""Host bit-model witness for the encoding premise; no GPU simulation claim."""

import json
import random
import struct
from fractions import Fraction
from pathlib import Path


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bf16(x):
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


rng = random.Random(19)
values = [0.0, 2.0**-110, -(2.0**-110), 1.0, 1 + 2.0**-23, 2**127]
for bits in [0x7F7F7FFF, 0xFF7F7FFF, 0x08800001, 0x88800001]:
    values.append(struct.unpack("<f", struct.pack("<I", bits))[0])
for _ in range(12000):
    exponent = rng.randrange(17, 254)
    bits = (rng.randrange(2) << 31) | (exponent << 23) | rng.randrange(1 << 23)
    values.append(struct.unpack("<f", struct.pack("<I", bits))[0])
for x in values:
    hi = bf16(x)
    rem = f32(x - hi)
    mid = bf16(rem)
    lo = bf16(f32(rem - mid))
    assert sum(map(Fraction, (hi, mid, lo))) == Fraction(x), x
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "outcome": "passed",
            "cases": len(values),
            "scope": (
                "CPU nearest-even bit model with gradual underflow, finite selecte"
                "d exponents; not native evidence"
            ),
        },
        indent=2,
    )
    + "\n"
)
