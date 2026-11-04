"""Logical incidence and bounded queue witnesses; no transport simulation."""

import itertools
import json
from pathlib import Path

cases = 0
for n in range(6):
    for dest in itertools.combinations(range(8), n):
        classes = {}
        for d in dest:
            classes.setdefault(d // 2, []).append(d)
        assert sorted(d for ds in classes.values() for d in ds) == list(dest)
        assert len(classes) == len(set(d // 2 for d in dest))
        cases += 1
queue_cases = 0
for width in range(1, 5):
    for sizes in itertools.product(range(width + 1), repeat=5):
        pending, issued, total = [], [], 0
        for n in sizes:
            pending.extend(range(total, total + n))
            total += n
            assert len(pending) <= 2 * width
            if len(pending) > width:
                issued.extend(pending[:width])
                pending = pending[width:]
            assert len(pending) <= width
        assert bool(pending) == bool(total)
        issued.extend(pending)
        assert issued == list(range(total))
        queue_cases += 1
Path(__file__).with_name("replay.json").write_text(
    json.dumps(
        {
            "quotient_cases": cases,
            "queue_cases": queue_cases,
            "outcome": "passed",
            "domain": (
                "unique destinations0..7 up to5; widths1..4, five rounds; no device execution"
            ),
        },
        indent=2,
    )
    + "\n"
)
