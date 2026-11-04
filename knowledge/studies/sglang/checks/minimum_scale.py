"""Finite UE8M0 scale identity, including its one admitted FP32 subnormal."""

import json
import math
import struct
from pathlib import Path

cases = []
for code in range(255):
    scale = math.ldexp(1.0, code - 127)
    bits = struct.unpack("I", struct.pack("f", scale))[0]
    encoded = 0 if scale == math.ldexp(1.0, -127) else bits >> 23 & 255
    assert encoded == code and math.ldexp(1.0, encoded - 127) == scale
    cases.append(code)
minimum_bits = struct.unpack("I", struct.pack("f", math.ldexp(1.0, -127)))[0]
assert minimum_bits >> 23 == 0 and minimum_bits & 0x7FFFFF
assert math.ldexp(1.0, -127) > 0
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "passed": True,
            "finite_codes_checked": len(cases),
            "minimum_is_positive_subnormal": True,
            "zero_is_not_a_live_scale": True,
            "scope": (
                "Exact host FP32/biased-exponent identity only; no native E8M0 rou"
                "nding/FP8 conversion claim"
            ),
        },
        indent=2,
    )
    + "\n"
)
