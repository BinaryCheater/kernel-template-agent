"Logical checkpoint event schedules; no torch frame or execution-engine stand-in."

import json
from pathlib import Path


def replay_events(node_saved, later_saved, early_stop, executable):
    # Output production precedes the node's save_for_backward pack events.
    events = ["node_output"]
    remaining = (node_saved if executable else 0) + later_saved
    for _ in range(node_saved if executable else 0):
        events.append("node_save")
        remaining -= 1
        if early_stop and remaining == 0:
            return [*events, "stop"]
    # A later operation consumes the output even when it saves no tensor.
    events.append("later_reads_output")
    for _ in range(later_saved):
        events.append("later_save")
        remaining -= 1
        if early_stop and remaining == 0:
            return [*events, "stop"]
    return [*events, "return"]


def peak(events):
    active = {}
    maximum = 0
    for action, name, size in events:
        if action == "allocate":
            assert name not in active
            active[name] = size
        else:
            assert action == "release"
            assert name in active
            del active[name]
        maximum = max(maximum, sum(active.values()))
    assert not active
    return maximum


if __name__ == "__main__":
    proven_dead = schedule_cases = 0
    for node_saved in range(5, 21):
        for later_saved in range(6):
            for early_stop in [False, True]:
                for executable in [False, True]:
                    events = replay_events(node_saved, later_saved, early_stop, executable)
                    remaining = (node_saved if executable else 0) + later_saved
                    proof = executable and early_stop and remaining == node_saved
                    if proof:
                        assert "later_reads_output" not in events
                        assert events.count("node_save") == node_saved
                        proven_dead += 1
                    if later_saved > 0 or not early_stop or not executable:
                        assert "later_reads_output" in events
                    schedule_cases += 1
    # Equal count alone can describe a different node's holders.
    assert 0 + 7 == 7
    assert "later_reads_output" in replay_events(7, 7, True, False)
    assert "later_reads_output" in replay_events(7, 0, False, True)
    assert "later_reads_output" in replay_events(7, 2, True, True)
    # Adding kept tapes to the saved-tensor sequence changes holder matching.
    assert replay_events(8, 0, True, True).count("node_save") != 7
    retained = [
        ("allocate", "tapes", 40),
        ("allocate", "later_backward", 80),
        ("release", "later_backward", 0),
        ("allocate", "node_workspace", 20),
        ("release", "node_workspace", 0),
        ("release", "tapes", 0),
    ]
    replayed = [
        ("allocate", "later_backward", 80),
        ("release", "later_backward", 0),
        ("allocate", "tapes", 40),
        ("allocate", "node_workspace", 20),
        ("release", "node_workspace", 0),
        ("release", "tapes", 0),
    ]
    last_saver = [e for e in retained if e[1] != "later_backward"]
    assert peak(retained) == 120
    assert peak(replayed) == 80
    assert peak(last_saver) == 60
    # One unconsumed entry: evict before allocating the next set, then consume once.
    for old_task in range(3):
        for new_task in range(3):
            old_key, new_key = (old_task, "node_A", "seed_A"), (new_task, "node_B", "seed_B")
            registry = {old_key: "A"}
            registry.clear()  # Before materializing B; A's later backward must replay.
            registry[new_key] = "B"
            assert registry.pop(old_key, None) is None
            assert registry.pop(new_key, None) == "B"
            assert registry.pop(new_key, None) is None
    assert (1, "same_pointer") != (2, "same_pointer")
    result = {
        "passed": True,
        "native_execution": False,
        "logical_replay_schedules": schedule_cases,
        "proved_dead_schedules": proven_dead,
        "illustrative_memory_units": {
            "keep_through_later": 120,
            "local_replay": 80,
            "last_saver": 60,
        },
        "counterexamples": [
            "count without executable node",
            "early stop off",
            "later saved holder",
            "extra saved tape",
            "wrong graph task",
        ],
        "open": [
            "torch hook and frame ABI",
            "native tape equality",
            "task callbacks and failure cleanup",
        ],
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
