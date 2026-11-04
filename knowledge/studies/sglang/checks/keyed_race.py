"Declared hash-selector endpoints and partition equivalence, not RNG statistics."

import json
import math
import struct
from pathlib import Path


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


endpoints = []
for h in (0, 1, 2**31 - 1, 2**31, 2**32 - 129, 2**32 - 128, 2**32 - 2, 2**32 - 1):
    naive = f32(f32(f32(h) + 0.5) / 2**32)
    logu = (
        math.log((h + 0.5) / 2**32) if h < 2**31 else math.log1p(-((2**32 - 1 - h) + 0.5) / 2**32)
    )
    assert logu < 0 and math.isfinite(logu)
    endpoints.append({"h": h, "naive_u_f32": naive, "complement_log": logu})
cases = 0
# Deliberately include score ties; use original identity even when rank order changes.
for length in range(1, 65):
    records = [(-float((i * 17) % 11), i) for i in range(length)]
    records = records[::2] + records[1::2]

    def cmp(r):
        return (r[0], -r[1])

    expected = max(records, key=cmp)
    for width in (1, 2, 3, 8, 17, 64):
        partial = [max(records[i : i + width], key=cmp) for i in range(0, length, width)]
        assert max(partial, key=cmp) == expected
        cases += 1
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "passed": True,
            "partition_cases": cases,
            "endpoints": endpoints,
            "scope": (
                "Host score comparator/endpoint algebra only; no hash quality, GPU"
                " or distribution claim"
            ),
        },
        indent=2,
    )
    + "\n"
)
