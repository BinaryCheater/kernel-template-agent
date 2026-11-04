"Logical first-input record ownership, not native append/quantization execution."

import itertools
import json
from pathlib import Path

if __name__ == "__main__":
    cases = 0
    for n in range(6):
        for slots in itertools.product([-1, 0, 1, 3], repeat=n):
            scan = {s: i for i, s in reversed(list(enumerate(slots))) if 0 <= s < 3}
            by_scan = {
                s: i
                for i, s in enumerate(slots)
                if 0 <= s < 3 and not any(x == s for x in slots[:i])
            }
            assert scan == by_scan
            for order in [range(n), reversed(range(n))]:
                owners = [0xFFFFFFFF] * 3
                for i in order:
                    if 0 <= slots[i] < 3:
                        owners[slots[i]] = min(owners[slots[i]], i)
                winners = {s: i for i, s in enumerate(slots) if 0 <= s < 3 and owners[s] == i}
                assert winners == scan
            cases += 1
    slots = [1, -1, 1, 0, 1]
    owners_seen = set()
    for order in itertools.permutations(range(5)):
        owners = [0xFFFFFFFF] * 2
        for i in order:
            if slots[i] >= 0:
                owners[slots[i]] = min(owners[slots[i]], i)
        owners_seen.add(tuple(owners))
    assert owners_seen == {(3, 0)}
    payloads = [(i * 100, i * 10 + 1, i * 10 + 2) for i in range(5)]
    winner_record = payloads[0]
    torn_record = (payloads[0][0], payloads[4][1], payloads[2][2])
    assert torn_record not in payloads and torn_record != winner_record
    # A prior invocation's smaller owner would suppress the new first occurrence.
    new_slots = [-1, 0]
    assert min(0, 1) != 1 and new_slots[1] == 0
    result = {
        "kind": "logical_first_occurrence_record_scatter",
        "native_execution": False,
        "command": (
            ".venv/bin/python knowledge/studies/flashinfer/checks/check_record_election.py"
        ),
        "slot_map_cases": cases,
        "all_claim_orders": 120,
        "winner_ids_for_slots_0_1": [3, 0],
        "whole_winner_record": winner_record,
        "torn_record": torn_record,
        "stale_owner_can_suppress_new_first_occurrence": True,
        "remaining": (
            "Native warp vote, atomics/pass publication, cache encoder/stride "
            "mapping, owner-padding restoration and consumer visibility"
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
