"""Step working-set pinning and inverse-map identity; no GPU or I/O test."""

import json
from itertools import product
from pathlib import Path

cases = 0
for K in range(1, 4):
    E = 4
    for resident in product(range(-1, E), repeat=K):
        if len({e for e in resident if e >= 0}) != sum(e >= 0 for e in resident):
            continue
        for T in range(K + 1):
            for routes in product(range(-1, E), repeat=T):
                slots = list(resident)
                inverse = [-1] * E
                for s, e in enumerate(slots):
                    if e >= 0:
                        inverse[e] = s
                age = list(range(K))
                now = K + 10
                for e in routes:
                    if e >= 0 and inverse[e] >= 0:
                        age[inverse[e]] = now
                plan = []
                for e in routes:
                    if e < 0 or inverse[e] >= 0:
                        continue
                    candidates = [s for s in range(K) if age[s] != now]
                    assert candidates
                    s = min(candidates, key=lambda s: (age[s], s))
                    old = slots[s]
                    if old >= 0:
                        inverse[old] = -1
                    slots[s] = e
                    inverse[e] = s
                    age[s] = now
                    plan.append((e, s))
                assert len({e for e, s in plan}) == len(plan)
                for s, e in enumerate(slots):
                    if e >= 0:
                        assert inverse[e] == s
                for e in routes:
                    if e >= 0:
                        assert slots[inverse[e]] == e and age[inverse[e]] == now
                cases += 1
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "passed": True,
            "cases": cases,
            "scope": (
                "Finite host cache maps, duplicate routes, protected working sets "
                "and LRU tie order; no native payload publication or GPU copy clai"
                "m"
            ),
        },
        indent=2,
    )
    + "\n"
)
