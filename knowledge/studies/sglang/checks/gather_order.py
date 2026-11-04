"Unique destinations and duplicate source IDs distinguish sort from deduplication."

import itertools
import json
from pathlib import Path

cases = 0
for n in range(8):
    for ids in itertools.product(range(3), repeat=n):
        perm = sorted(range(n), key=lambda i: ids[i])
        output = [None] * n
        for j in perm:
            output[j] = (ids[j], ids[j] + 10, ids[j] + 20)
        assert output == [(i, i + 10, i + 20) for i in ids]
        assert sorted(perm) == list(range(n))
        cases += 1
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "outcome": "passed",
            "cases": cases,
            "domain": "length0..7, source IDs0..2; unique coordinate tuples",
            "scope": "Permutation and destination identity only",
        },
        indent=2,
    )
    + "\n"
)
