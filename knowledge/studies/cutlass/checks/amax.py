"""Replay finite pre-conversion amax and its signed-bit atomic implementation."""

from __future__ import annotations

import itertools
import json
import math
import struct
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def signed(value: int) -> int:
    return value if value < 2**31 else value - 2**32


def number(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value))[0]


def main() -> None:
    domains = 0
    # val has known zero sign: the source atomic uses signed Int32 max, not Uint32 max.
    for initial, update in itertools.product(
        (-math.inf, -10.0, -0.0, 0.0, 1.0, math.inf), (0.0, 0.5, 2.0, 2049.0, math.inf)
    ):
        result = number(max((bits(initial), bits(update)), key=signed))
        assert result == max(initial, update)
        domains += 1
    for values in itertools.product((0.0, 0.5, 2.0, 2049.0), repeat=4):
        for order in itertools.permutations(range(4)):
            accumulator = bits(-math.inf)
            for i in order:
                accumulator = max((accumulator, bits(values[i])), key=signed)
            assert number(accumulator) == max(values)
            domains += 1
    assert number(max(bits(-math.inf), bits(2.0))) == -math.inf
    original = 2049.0
    converted = struct.unpack("<e", struct.pack("<e", original))[0]
    assert converted == 2048.0 and max(abs(original), 0.0) != abs(converted)
    # A fresh smaller invocation without resetting the scalar keeps an older maximum.
    stale = number(max((bits(9.0), bits(2.0)), key=signed))
    assert stale == 9.0
    result = {
        "outcome": "PASS",
        "finite_order_domains": domains,
        "counterexamples": {
            "unsigned_max_with_negative_infinity": "-inf",
            "preconversion_amax": original,
            "postconversion_amax": converted,
            "unreset_next_invocation": stale,
        },
        "boundary": (
            "Nonnegative non-NaN update model; NaN policy, reduction lanes and GPU atomicity open."
        ),
    }
    (STUDY / "checks/amax-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: finite signed-bit max order, preconversion witness and reset premise.\n"
    )


if __name__ == "__main__":
    main()
