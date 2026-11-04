"""Replay contiguous scheduling and sparse merge-path metadata equations."""

from __future__ import annotations

import bisect
import itertools
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
INVALID = 65535


def merge_path(
    a: tuple[int, ...],
    b: tuple[int, ...],
    threads: int = 256,
    *,
    drop_carried_duplicate: bool = True,
) -> list[tuple[int, int, bool, int, bool]]:
    merged = []
    total = len(a) + len(b)
    for thread in range(threads):
        begin, end = thread * total // threads, (thread + 1) * total // threads
        lo, hi = max(begin, len(b)) - len(b), min(begin, len(a))
        while lo < hi:
            ai = (lo + hi) // 2
            bi = begin - ai
            if bi > 0 and ai < len(a) and b[bi - 1] >= a[ai]:
                lo = ai + 1
            else:
                hi = ai
        ai, bi, remaining = lo, begin - lo, end - begin
        if (
            drop_carried_duplicate
            and remaining > 0
            and ai > 0
            and bi < len(b)
            and a[ai - 1] == b[bi]
        ):
            bi += 1
            remaining -= 1
        while remaining:
            av = a[ai] if ai < len(a) else 0xFFFFFFFF
            bv = b[bi] if bi < len(b) else 0xFFFFFFFF
            in_a, in_b = av <= bv, bv <= av
            consume_b = in_b and remaining > in_a
            merged.append((min(av, bv), ai, in_a, bi, in_b))
            ai += in_a
            bi += consume_b
            remaining -= in_a + consume_b
    return merged


def scheduled_splits(counts: tuple[int, ...], sms: int) -> list[list[tuple[int, int]]]:
    work, cost = [], []
    for count in counts:
        work.append((work[-1] if work else 0) + count)
        cost.append((cost[-1] if cost else 0) + 2 * count + (3 if count else 0))
    total_cost, total_work = cost[-1], work[-1]

    def locate(boundary: int) -> tuple[int, int, int]:
        target = boundary * (total_cost // sms) + min(boundary, total_cost % sms)
        if target == total_cost:
            return len(counts), 0, total_work
        q = bisect.bisect_right(cost, target)
        cost_before, work_before = (cost[q - 1], work[q - 1]) if q else (0, 0)
        offset = min((max(target - cost_before, 3) - 3) // 2, counts[q] - 1)
        return q, offset, work_before + offset

    all_tasks = []
    for sm in range(sms):
        q, offset, start_work = locate(sm)
        remaining = locate(sm + 1)[2] - start_work
        tasks = []
        while remaining and q < len(counts):
            if offset < counts[q]:
                take = min(counts[q] - offset, remaining)
                tasks.extend((q, i) for i in range(offset, offset + take))
                remaining -= take
            offset = 0
            q += 1
        assert remaining == 0
        all_tasks.append(tasks)
    return all_tasks


def main() -> None:
    root = Path(json.loads((STUDY / "inputs/sources.json").read_text())["tree"])
    source = (
        root / "deep_gemm/include/deep_gemm/scheduler/sm100_sparse_mqa_logits_metadata.cuh"
    ).read_text()
    for marker in (
        "Drop a duplicate carried across merge partitions",
        "num_remaining_inputs > in_q0",
        "q0_slot_idx - q0_slot_base",
        "kInvalidSparseSlot",
        "last_logical_kv_block_idx == logical_kv_block_idx",
    ):
        assert marker in source
    subsets = [tuple(i for i in range(5) if mask & (1 << i)) for mask in range(32)]
    merge_cases = 0
    for a, b in itertools.product(subsets, repeat=2):
        merged = merge_path(a, b)
        assert [entry[0] for entry in merged] == sorted(set(a) | set(b))
        for width in (1, 2, 3, 4, 8):
            recovered = [dict(), dict()]
            for start in range(0, len(merged), width):
                chunk = merged[start : start + width]
                bases = [chunk[0][1], chunk[0][3]]
                padded = chunk + [(0, 0, False, 0, False)] * (width - len(chunk))
                for key, ai, in_a, bi, in_b in padded:
                    for row, idx, present in ((0, ai, in_a), (1, bi, in_b)):
                        offset = idx - bases[row] if present else INVALID
                        if offset != INVALID:
                            slot = bases[row] + offset
                            assert slot not in recovered[row]
                            recovered[row][slot] = key
            assert [recovered[row] for row in range(2)] == [dict(enumerate(a)), dict(enumerate(b))]
        merge_cases += 1
    assert len(merge_path((2,), (2,), drop_carried_duplicate=False)) == 2
    assert len(merge_path((2,), (2,))) == 1
    schedule_cases = 0
    for counts in itertools.product((0, 1, 2, 5), repeat=4):
        for sms in (1, 2, 3, 7, 132):
            actual = list(itertools.chain.from_iterable(scheduled_splits(counts, sms)))
            expected = [(q, i) for q, count in enumerate(counts) for i in range(count)]
            assert sorted(actual) == expected and len(actual) == len(set(actual))
            schedule_cases += 1
    # Sorted strict uniqueness is necessary for checking contiguity by endpoints alone.
    assert [0, 1, 1, 3][-1] == [0, 1, 1, 3][0] + 4 - 1
    assert len(set([0, 1, 1, 3])) != 4
    # Same-chunk reuse preserves ring identity only after exactly one full rotation.
    reuse_domains = []
    for stages in (3, 8):
        accepted = []
        for count in range(1, 2 * stages + 1):
            slots = [None] * stages
            cursor = 2 % stages
            for chunk in range(count):
                slots[cursor] = chunk
                cursor = (cursor + 1) % stages
            replay = [slots[(cursor + chunk) % stages] for chunk in range(count)]
            if replay == list(range(count)):
                accepted.append(count)
        assert accepted == [stages]
        reuse_domains.append({"stages": stages, "safe_same_chunk_lengths": accepted})
    # SM90 reversed allocation still yields a tail cursor when all context lengths are zero.
    empty_prefix = [0, 0, 0]
    empty_q_cursor = bisect.bisect_right(empty_prefix, 0)
    assert empty_q_cursor == len(empty_prefix)
    result = {
        "outcome": "PASS",
        "merge_path_row_pairs": merge_cases,
        "split_widths": [1, 2, 3, 4, 8],
        "full_ring_reuse_domains": reuse_domains,
        "sm90_all_empty_context_cursor": {
            "request_count": 3,
            "initial_cursor": empty_q_cursor,
            "source_refresh_would_be_in_bounds": False,
        },
        "scheduled_count_domains": schedule_cases,
        "mutations_rejected": [
            "keep_cross_partition_duplicate",
            "endpoint_contiguity_without_uniqueness",
        ],
        "shared_union_witness": merge_path((0, 2, 4), (1, 2, 4)),
        "boundary": (
            "Source merge/schedule equations and scatter identity; no GPU scan"
            " or atomic publication."
        ),
    }
    (STUDY / "checks/attention-metadata-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    sys.stdout.write(
        f"PASS: {merge_cases} merge pairs and {schedule_cases} scheduled-count domains.\n"
    )


if __name__ == "__main__":
    main()
