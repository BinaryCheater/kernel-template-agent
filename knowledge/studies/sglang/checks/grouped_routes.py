"Incidence identity and independent padded origins, not device atomic ordering."

import json
from bisect import bisect_left
from itertools import product
from pathlib import Path

cases = 0
for n in range(7):
    for keys in product((-1, 0, 1, 2), repeat=n):
        counts = [keys.count(e) for e in range(3)]
        offsets = [0]
        for c in counts:
            offsets.append(offsets[-1] + c)
        for order in (list(range(n)), list(reversed(range(n)))):
            cursors = offsets[:-1].copy()
            gather = [None] * offsets[-1]
            inverse = [-1] * n
            for r in order:
                e = keys[r]
                if e < 0:
                    continue
                d = cursors[e]
                cursors[e] += 1
                gather[d] = r // 2
                inverse[r] = d
            assert len({d for d in inverse if d >= 0}) == offsets[-1]
            for r, e in enumerate(keys):
                if e >= 0:
                    assert (
                        offsets[e] <= inverse[r] < offsets[e + 1] and gather[inverse[r]] == r // 2
                    )
                else:
                    assert inverse[r] < 0
        reorder = sorted(range(n), key=lambda r: keys[r])
        sorted_keys = [keys[r] for r in reorder]
        neg = bisect_left(sorted_keys, 0)
        sorted_offsets = [bisect_left(sorted_keys, e) - neg for e in range(3)] + [n - neg]
        assert sorted_offsets == offsets
        inv2 = [None] * n
        for i, r in enumerate(reorder):
            inv2[r] = i - neg
        assert all((inv2[r] < 0) == (e < 0) for r, e in enumerate(keys))
        # Data and MX scale rows need separate prefixes; repeated token IDs remain incidences.
        scale = [0]
        for c in counts:
            scale.append(scale[-1] + ((c + 127) // 128) * 128)
        for e, c in enumerate(counts):
            assert scale[e + 1] - scale[e] >= c
        cases += 1
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "passed": True,
            "cases": cases,
            "scope": (
                "Exact finite incidence maps, two reservation orders, sorted bound"
                "s and separate128-row scale origins; no GPU synchronization or nu"
                "merical timing claim"
            ),
        },
        indent=2,
    )
    + "\n"
)
