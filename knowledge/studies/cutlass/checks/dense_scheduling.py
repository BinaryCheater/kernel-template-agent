"Check cancellation ownership, split-K mailbox incidence and source IR guard gap."

from __future__ import annotations

import ast
import collections
import itertools
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def cancellation_states(n: int) -> int:
    initial = (tuple(range(n)), (), ())
    queue, seen = collections.deque([initial]), {initial}
    while queue:
        pending_tuple, active_tuple, done_tuple = queue.popleft()
        pending, active, done = set(pending_tuple), dict(active_tuple), set(done_tuple)
        assert len(set(active.values())) == len(active)
        assert not (pending & done or pending & set(active.values()) or done & set(active.values()))
        assert pending | set(active.values()) | done == set(range(n))
        next_states = []
        for task in pending:
            # Hardware dispatch starts an uncancelled launch group once.
            next_states.append((pending - {task}, {**active, task: task}, done))
        for worker, task in active.items():
            remaining = {k: v for k, v in active.items() if k != worker}
            # A failed request retires this worker; pending tasks can still launch natively.
            next_states.append((pending, remaining, done | {task}))
            for stolen in pending:
                next_states.append(
                    (pending - {stolen}, {**remaining, worker: stolen}, done | {task})
                )
        if not next_states:
            assert done == set(range(n))
        for p, a, d in next_states:
            state = (tuple(sorted(p)), tuple(sorted(a.items())), tuple(sorted(d)))
            if state not in seen:
                seen.add(state)
                queue.append(state)
    return len(seen)


def mailbox_domains() -> dict[str, int]:
    count = coordinates = empty_domains = 0
    for split, per_thread, ktiles in itertools.product((2, 4, 8), (8, 16, 32), range(17)):
        shard = per_thread // split
        ktile_chunk = (ktiles + split - 1) // split
        segments = [
            range(min(rank * ktile_chunk, ktiles), min((rank + 1) * ktile_chunk, ktiles))
            for rank in range(split)
        ]
        assert [x for segment in segments for x in segment] == list(range(ktiles))
        empty_domains += any(not segment for segment in segments)
        slots = {}
        for receiver, sender, thread, i in itertools.product(
            range(split), range(split), range(128), range(shard)
        ):
            if receiver == sender:
                continue
            sender_slot = sender - (sender > receiver)
            address = thread + 128 * i + 128 * shard * sender_slot
            key = receiver, address
            assert key not in slots
            # Identify the sender's value for the receiver-owned output shard.
            slots[key] = (sender, thread, receiver * shard + i)
            coordinates += 1
        for receiver, thread, i in itertools.product(range(split), range(128), range(shard)):
            contributions = [(receiver, thread, receiver * shard + i)]
            contributions += [
                slots[(receiver, thread + 128 * i + 128 * shard * s)] for s in range(split - 1)
            ]
            assert {x[0] for x in contributions} == set(range(split))
            assert {x[2] for x in contributions} == {receiver * shard + i}
        expected_bytes = (split - 1) * 128 * shard * 4
        assert expected_bytes == len([key for key in slots if key[0] == 0]) * 4
        count += 1
    return {
        "domains": count,
        "mailbox_coordinates": coordinates,
        "domains_with_empty_k_partition": empty_domains,
    }


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    clc = ast.parse((root / "python/CuTeDSL/cutlass/cute/arch/clc.py").read_text())
    response = next(
        x for x in clc.body if isinstance(x, ast.FunctionDef) and x.name == "clc_response"
    )
    unguarded = []
    for statement in response.body:
        for node in ast.walk(statement):
            if isinstance(statement, ast.If):
                continue
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "query_type" and isinstance(keyword.value, ast.Constant):
                        if str(keyword.value.value).startswith("get_first_cta_id"):
                            unguarded.append(keyword.value.value)
    assert len(unguarded) == 3
    result = {
        "outcome": "PASS",
        "cancellation_model_states": {str(n): cancellation_states(n) for n in range(1, 6)},
        "split_k": mailbox_domains(),
        "empty_partition_witness": {
            "k_tiles": 5,
            "splits": 4,
            "ranges": [[0, 2], [2, 4], [4, 5], [5, 5]],
            "source": "No identity accumulator write observed for zero-iteration MMA loop.",
        },
        "source_finding": {
            "unguarded_coordinate_query_ir": unguarded,
            "status": "Guarded lowering required; no safe source realization claimed.",
        },
        "boundary": (
            "Bounded ownership/incidence models; hardware cancellation and DSMEM ordering open."
        ),
    }
    (STUDY / "checks/dense-scheduling-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: cancellation ownership and split-K mailboxes; source query-guard gap retained.\n"
    )


if __name__ == "__main__":
    main()
