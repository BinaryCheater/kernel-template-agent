"Check selected-group expansion with unique coordinates, not low-bit repeating values."

import json
from pathlib import Path

cases = 0
for groups in range(1, 65):
    for repair in range(groups + 1):
        owners = {}
        for g in range(groups):
            pos = g + max(0, g - (groups - repair))
            assert pos not in owners
            owners[pos] = (g, "base")
            if g >= groups - repair:
                assert pos + 1 not in owners
                owners[pos + 1] = (g, "residual")
        assert sorted(owners) == list(range(groups + repair))
        assert len({g for g, kind in owners.values() if kind == "residual"}) == repair
        cases += 1
Path(__file__).with_name("replay.json").write_text(
    json.dumps(
        {
            "outcome": "passed",
            "cases": cases,
            "domain": "G=1..64, R=0..G, unique group/role identities",
            "scope": "Logical expansion only; no native numeric conversion or consumer execution",
        },
        indent=2,
    )
    + "\n"
)
