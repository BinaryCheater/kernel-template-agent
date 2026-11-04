"""Check all packed E2M1 byte predicates; signed-zero bits are not magnitude."""

import json
from pathlib import Path

zero = []
for byte in range(256):
    expected = (byte & 7) == 0 and ((byte >> 4) & 7) == 0
    assert ((byte & 0x77) == 0) == expected
    if expected:
        zero.append(byte)
assert zero == [0, 8, 128, 136]
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "outcome": "passed",
            "bytes_tested": 256,
            "zero_pair_bytes": zero,
            "scope": (
                "Packed magnitude predicate only; native copy/MMA pairing and IEEE"
                " corner cases remain separate"
            ),
        },
        indent=2,
    )
    + "\n"
)
