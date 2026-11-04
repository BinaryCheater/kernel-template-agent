"""Local probe, sketch, bitmap and buffer witnesses; no GPU execution."""

import json
import sys

# Full-cycle condition separates legal double hash from a plausible invalid one.
assert len({(2 + i * 3) % 7 for i in range(7)}) == 7
assert len({(2 + i * 2) % 8 for i in range(8)}) == 4
# Tombstone preserves a collision chain; empty would end it.
slots = ["erased", "B", "empty"]
assert next(x for x in slots if x != "erased") == "B"
# Shared retrieval capacity must admit one whole next step before append.
buffer, step, pending = 96, 64, 40
assert pending > buffer - step and pending + step > buffer
# Register max merge, not histogram addition.
a, b = [0, 5, 1], [2, 3, 1]
assert [max(x, y) for x, y in zip(a, b, strict=False)] == [2, 5, 1]
# Prefix reconstruction and inclusive run endpoint with asymmetric coordinates.
value = 0x12340007
assert (value >> 16, value & 65535) == (0x1234, 7)
assert 5 <= 7 <= 5 + 2 and not (5 <= 8 <= 5 + 2)
# Bloom OR must retain independent concurrent insertion effects.
assert (1 << 0) | (1 << 3) == 9
sys.stdout.write(
    " ".join(
        str(value)
        for value in [
            json.dumps(
                {
                    "prime_probe_cycle": 7,
                    "invalid_composite_cycle": 4,
                    "shared_flush_before_append": True,
                    "max_merged": [2, 5, 1],
                    "inclusive_run_endpoint": True,
                    "scope": "local algebra and indexing; no concurrency/accuracy result",
                },
                indent=2,
            )
        ]
    )
    + "\n"
)
