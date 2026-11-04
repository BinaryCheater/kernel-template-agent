"Explore the extracted logical barrier, not CUDA memory-model implementation."

import json
import sys
from collections import deque
from pathlib import Path


def explore(counter_banks: int, data_banks: int) -> dict[str, object]:
    peers, rounds = 2, 8
    # Each participant executes write -> release arrival -> acquire and read.
    initial = ((0, 0), (0, 0), (0,) * counter_banks, (-1,) * (data_banks * peers))
    pending = deque([(initial, [])])
    seen = {initial}
    completed = 0
    while pending:
        (generation, phase, counters, data), history = pending.popleft()
        if all(g == rounds for g in generation):
            completed += 1
            continue
        enabled = False
        for actor in range(peers):
            g = generation[actor]
            if g == rounds:
                continue
            p = phase[actor]
            cb, db = g % counter_banks, g % data_banks
            step = 1 if (g // counter_banks) % 2 == 0 else -1
            target = peers if step == 1 else 0
            if p == 2 and counters[cb] != target:
                continue
            enabled = True
            ng, np, nc, nd = map(list, (generation, phase, counters, data))
            event = f"actor {actor}, generation {g}, "
            if p == 0:
                nd[db * peers + actor] = g
                np[actor] = 1
                event += "write"
            elif p == 1:
                nc[cb] += step
                np[actor] = 2
                event += "release arrival"
            else:
                event += "acquire and read"
                if any(data[db * peers + j] != g for j in range(peers)):
                    return {"outcome": "stale_or_overwritten_read", "witness": [*history, event]}
                ng[actor] += 1
                np[actor] = 0
            state = tuple(map(tuple, (ng, np, nc, nd)))
            if state not in seen:
                seen.add(state)
                pending.append((state, [*history, event]))
        if not enabled:
            return {"outcome": "deadlock", "witness": history}
    return {"outcome": "safe_and_terminating", "states": len(seen), "terminal_states": completed}


if __name__ == "__main__":
    result = {
        "kind": "finite_abstract_interleaving_check",
        "domain": {"participants": 2, "generations": 8},
        "assumptions": [
            "release/acquire publishes leader-owned partials",
            "actors eventually execute enabled steps",
            "all participants follow identical generation order",
        ],
        "native_execution": False,
        "two_counters_two_data_banks": explore(2, 2),
        "one_counter_two_data_banks": explore(1, 2),
        "two_counters_one_data_bank": explore(2, 1),
    }
    assert result["two_counters_two_data_banks"]["outcome"] == "safe_and_terminating"
    assert result["one_counter_two_data_banks"]["outcome"] == "deadlock"
    assert result["two_counters_one_data_bank"]["outcome"] == "stale_or_overwritten_read"
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    summary = {
        key: value["outcome"]
        for key, value in result.items()
        if isinstance(value, dict) and "outcome" in value
    }
    sys.stdout.write(json.dumps(summary) + "\n")
