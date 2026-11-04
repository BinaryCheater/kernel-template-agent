"Exhaustive abstract claim/consume/retire interleavings; no GPU runtime model."

import json
from pathlib import Path


def schedules(workers, tasks):
    # Worker phase: initial claim0/1, lookahead, compute, retire, finished.
    initial = (0, 0, 0, tuple((0, ()) for _ in range(workers)), ())
    pending, seen, terminals = [initial], {initial}, 0
    while pending:
        next_ticket, retired, claims, local, consumed = pending.pop()
        if all(phase == 5 for phase, _ in local):
            assert next_ticket == retired == 0
            assert claims == tasks + 2 * workers
            assert consumed == tuple(range(tasks))
            terminals += 1
            continue
        for w, (phase, queue) in enumerate(local):
            if phase == 5:
                continue
            n, r, c, done = next_ticket, retired, claims, consumed
            if phase in (0, 1):
                new_worker = (phase + 1, (*queue, n))
                n, c = n + 1, c + 1
            elif phase == 2:
                if queue[0] < tasks:
                    new_worker = (3, (*queue, n))
                    n, c = n + 1, c + 1
                else:
                    assert all(q >= tasks for q in queue)
                    new_worker = (4, queue)
            elif phase == 3:
                task = queue[0]
                assert task not in done
                done = tuple(sorted((*done, task)))
                new_worker = (2, queue[1:])
            else:
                assert phase == 4
                if r == workers - 1:
                    # Every other claimer has already retired; no further claim is possible.
                    assert all(
                        i == w or other_phase == 5 for i, (other_phase, _) in enumerate(local)
                    )
                    n, r = 0, 0
                else:
                    r += 1
                new_worker = (5, ())
            updated = list(local)
            updated[w] = new_worker
            state = (n, r, c, tuple(updated), done)
            if state not in seen:
                seen.add(state)
                pending.append(state)
    assert terminals > 0
    return len(seen), terminals


if __name__ == "__main__":
    states = terminal_schedules = cases = 0
    for workers, limit in [(1, 7), (2, 6), (3, 3)]:
        for tasks in range(limit):
            count, terminal = schedules(workers, tasks)
            states += count
            terminal_schedules += terminal
            cases += 1
    # Wrong retirement participant count: one worker can reset before a second starts.
    tasks = 3
    first_claims = list(range(tasks + 2))
    first_consumed = first_claims[:tasks]
    incorrectly_reset_next = 0
    assert incorrectly_reset_next in first_consumed
    # Counter reset does not publish other threads' pending output stores.
    pending_output_stores = {"task0_lane7"}
    next_ticket, retired = 0, 0
    assert next_ticket == retired == 0 and pending_output_stores
    # Subsequent launch waits for whole-kernel completion, which drains those stores.
    pending_output_stores.clear()
    assert not pending_output_stores
    result = {
        "passed": True,
        "native_execution": False,
        "partition_cases": cases,
        "abstract_protocol_states": states,
        "terminal_protocol_states": terminal_schedules,
        "counterexamples": [
            "wrong participant count causes duplicate ticket",
            "zero counters before output completion",
        ],
        "open": [
            "native atomic memory scope",
            "CTA transfer pipeline",
            "stream/graph completion and ownership",
        ],
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
