"""Local bit/format witnesses, not GPU kernel execution."""

import json
import sys

checked = 0
for old in range(16):
    for new in range(16):
        packed = old | (new << 4)
        assert packed & 15 == old
        assert packed >> 4 == new
        checked += 1
# Consequential misuse: OR cannot replace an already populated field.
assert ((3 | (10 << 4)) | (2 << 4)) >> 4 != 2
# Integer-ratio input avoids hiding truncation with repeated low-bit patterns.
assert int((19 / 10) / 15 * 15) == 1
assert 0 + (1 + 0.5) * 15 / 15 == 1.5
assert 0 + (15 + 0.5) * 15 / 15 == 15.5
counts = [2, 0, 3]
prefix = []
for n in counts:
    prefix.append((prefix[-1] if prefix else 0) + n)
assert prefix == [2, 2, 5]
sys.stdout.write(
    " ".join(
        str(value)
        for value in [
            json.dumps(
                {
                    "packed_nibble_cases": checked,
                    "old_field_preserved": True,
                    "overwrite_counterexample": True,
                    "midpoint_policy_witnesses": 3,
                    "scope": (
                        "local algebra only; shuffle lowering, FP32 boundaries and concurr"
                        "ency remain open"
                    ),
                },
                indent=2,
            )
        ]
    )
    + "\n"
)
