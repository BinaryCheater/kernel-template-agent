"Logical sparse occurrence validity and gradient multiplicity, not GPU training."

import itertools
import json
from pathlib import Path

if __name__ == "__main__":
    cases = 0
    for n in range(6):
        for keys in itertools.product([-1, 0, 1, 4], repeat=n):
            for cut in range(n + 1):
                valid = [(s, k + 10) for s, k in enumerate(keys) if s < cut and 0 <= k < 3]
                packed = [k for _, k in valid] + [-1] * (n - len(valid))
                span = max((s + 1 for s, _ in valid), default=0)
                assert packed[: len(valid)] == [k for _, k in valid]
                assert span >= len(valid)
                before = {}
                after = {}
                for s, k in valid:
                    before[k] = before.get(k, 0) + s + 1
                for i, (s, _) in enumerate(valid):
                    k = packed[i]
                    after[k] = after.get(k, 0) + s + 1
                assert after == before
                cases += 1
    row, cut = [-1, 2, 2, 1], 2
    correct = [k for s, k in enumerate(row) if s < cut and k >= 0]
    wrong = [k for k in row if k >= 0][:cut]
    assert correct == [2] and wrong == [2, 2]
    # Repeated key2 has twice the logical mass of key1, even with equal scores.
    repeated = [2, 2, 1]
    values = {2: 9.0, 1: 0.0}
    full = sum(values[k] for k in repeated) / len(repeated)
    unique = sum(values[k] for k in set(repeated)) / len(set(repeated))
    assert full == 6 and unique == 4.5
    # Query segment length2 is the tail of key segment length4, key-base10.
    local_q = 0
    bound = 4 - 2 + local_q
    selected = [k + 10 for k in [0, 3, 2] if 0 <= k <= bound]
    assert selected == [10, 12]
    result = {
        "kind": "logical_stable_support_compaction",
        "native_execution": False,
        "command": (
            ".venv/bin/python knowledge/studies/flashinfer/checks/check_support_compaction.py"
        ),
        "slot_cut_cases": cases,
        "cut_then_pack": correct,
        "wrong_pack_then_cut": wrong,
        "duplicate_mass_output": full,
        "wrong_deduplicated_output": unique,
        "causal_global_keys": selected,
        "remaining": (
            "Native stable scan, metadata/generation publication, forward/back"
            "ward tile rounding and native additive gradient accumulation"
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
