"Witness that raw-sum offset correction differs from reconstructing the code sum."

import json
from fractions import Fraction as F
from pathlib import Path

x = [F(1, 10), F(1, 10), F(1)] + [F(0)] * 29
# Simple admitted rounding illustration; this is not a native Q8 quantizer emulator.
scale = F(1, 4)
q = [round(v / scale) for v in x]
raw = sum(x)
reconstructed = scale * sum(q)
assert raw != reconstructed
m = F(3, 7)
for owners in [1, 2, 4, 8]:
    assert sum([m * raw / owners] * owners) == m * raw
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "outcome": "passed",
            "raw_sum": str(raw),
            "code_reconstructed_sum": str(reconstructed),
            "owner_counts": [1, 2, 4, 8],
            "scope": (
                "Exact correction identity and domain distinction only; nativeQ8/h"
                "alf2 rounding separate"
            ),
        },
        indent=2,
    )
    + "\n"
)
