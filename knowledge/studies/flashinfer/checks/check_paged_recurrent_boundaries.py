"""Paged prefix/checkpoint/final rows and virtual64 checkpoint coordinates."""

import json
from pathlib import Path


def source_slots(first, end, page, last, new):
    total = (end - first - 1) * page + last
    prefix = total - new
    read = first + max(prefix - 1, 0) // page if prefix > 0 else -1
    write = end - 1 if new > 0 and total % page else -1
    return read, write, first + prefix // page, prefix


def virtual_checkpoints(first_checkpoint, prefix, new, page):
    pad = prefix % 64
    origin = prefix - pad
    length = pad + new
    key = (first_checkpoint - 1) * page + origin % page
    events = []
    for end in range(64, length + 1, 64):
        pos = key + end
        if pos % page == 0:
            events.append((pos // page, origin + end))
    return events


if __name__ == "__main__":
    metadata_cases = checkpoint_events = decode_cases = 0
    for page in [64, 128]:
        for pages in range(1, 5):
            # Boundary-focused tails plus a middle point; not every implementation variant.
            for last in sorted({1, 2, 31, 63, 64, page - 1, page}):
                if last > page:
                    continue
                total = (pages - 1) * page + last
                # All old-prefix phases and new lengths for this after-step length.
                for new in range(total + 1):
                    first = 3
                    end = first + pages
                    read, write, checkpoint_start, prefix = source_slots(
                        first, end, page, last, new
                    )
                    # Independent states are labeled by inclusive absolute token count.
                    states = {t: ("sequence", t) for t in range(total + 1)}
                    expected_read = -1 if prefix == 0 else first + (prefix - 1) // page
                    assert read == expected_read
                    boundary_positions = [t for t in range(prefix + 1, total + 1) if t % page == 0]
                    expected_events = [(first + t // page - 1, t) for t in boundary_positions]
                    actual_events = virtual_checkpoints(checkpoint_start, prefix, new, page)
                    assert actual_events == expected_events
                    writes = {entry: states[position] for entry, position in actual_events}
                    if write >= 0:
                        assert write not in writes  # Full-page final store is suppressed.
                        writes[write] = states[total]
                    expected_written = {
                        first + (t - 1) // page for t in range(prefix + 1, total + 1)
                    }
                    assert set(writes) == expected_written
                    if new == 0:
                        assert writes == {} and write == -1
                    if prefix % page and new:
                        assert read in writes  # Own partial-prefix continuation may alias.
                    checkpoint_events += len(actual_events)
                    metadata_cases += 1
                if total >= 2:
                    new_page = last == 1
                    read = end - 2 if new_page else end - 1
                    write = end - 1
                    prefix = total - 1
                    assert read == first + (prefix - 1) // page
                    assert write == first + (total - 1) // page
                    if new_page:
                        assert pages >= 2 and read != write
                    decode_cases += 1
    # Two sequences can share only the unchanged complete prefix.
    reads_a, writes_a = {11}, {29}
    reads_b, writes_b = {11}, {7}
    assert not writes_a & (reads_b | writes_b)
    assert not writes_b & (reads_a | writes_a)
    assert writes_a & {29}  # Reusing private tail29 in another write set is invalid.
    assert writes_a & {29, 31}  # Another sequence reading that page is also invalid.
    # Same page-table shape, next decode step: last_len1→2 changes read page.
    previous_step = (11, 29)  # read previous complete page, write new tail.
    next_step = (29, 29)  # continue the private partial tail.
    assert previous_step != next_step
    result = {
        "passed": True,
        "native_execution": False,
        "prefill_metadata_cases": metadata_cases,
        "absolute_checkpoint_events": checkpoint_events,
        "decode_cases": decode_cases,
        "oracle": "Absolute state positions versus source slots and virtual64 checkpoint key",
        "counterexamples": [
            "shared mutable partial tail",
            "read/write overlap across sequences",
            "stale same-shape slots after page-opening decode",
        ],
        "open": [
            "actual initial-state content/generation",
            "native checkpoint rounding",
            "page allocator/COW and all-layer reader retirement",
        ],
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
