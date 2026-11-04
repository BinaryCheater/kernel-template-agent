"""Source-index and tagged-counter replays for hyperconnection and Mega MHC."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
MASK64 = (1 << 64) - 1


def partition(blocks: int, splits: int) -> list[range]:
    size, remainder = divmod(blocks, splits)
    return [
        range(i * size + min(i, remainder), i * size + min(i, remainder) + size + (i < remainder))
        for i in range(splits)
    ]


def check_split_policy() -> dict[str, object]:
    cases = 0
    for blocks, mblocks, sms in itertools.product(
        range(16, 257, 16), (1, 2, 3, 17), (1, 8, 32, 132)
    ):
        maximum = min(64, max(16, sms // mblocks))
        shortest_longest = (blocks + maximum - 1) // maximum
        splits = (blocks + shortest_longest - 1) // shortest_longest
        parts = partition(blocks, splits)
        assert [k for part in parts for k in part] == list(range(blocks))
        assert all(parts) and max(map(len, parts)) - min(map(len, parts)) <= 1
        assert max(map(len, parts)) == shortest_longest
        assert all(
            (blocks + candidate - 1) // candidate > shortest_longest
            for candidate in range(1, splits)
        )
        cases += 1
    # Standalone prenorm's public split count does not exclude empty producer tasks.
    assert list(map(len, partition(1, 2))) == [1, 0]
    return {
        "domains": cases,
        "unequal_witness": [list(p) for p in partition(17, 4)],
        "host_policy_witness": {
            "hidden": 7168,
            "num_sms": 132,
            "mblocks": 1,
            "blocks": 112,
            "selected_splits": 56,
        },
        "standalone_empty_split_witness": {"K": 64, "splits": 2, "block_counts": [1, 0]},
    }


def check_epoch() -> dict[str, object]:
    domains = 0
    for epoch, count in itertools.product((0, 1, 2, 7, (1 << 57) - 2), (1, 2, 3, 16, 64)):
        base = (epoch + 1) * 128
        assert base + count <= MASK64
        for current in range(count):
            assert ((base + current - base) & MASK64) < count
        assert not (((base + count - base) & MASK64) < count)
        for old_arrivals in range(65):
            old_state = epoch * 128 + old_arrivals if epoch else old_arrivals
            assert ((old_state - base) & MASK64) >= count
            assert old_state != base + count
        domains += 1
    arrival_orders = 0
    for count in (1, 2, 3):
        events = [(kind, producer) for producer in range(count) for kind in ("write", "arrive")]
        for order in itertools.permutations(events):
            if any(order.index(("write", i)) > order.index(("arrive", i)) for i in range(count)):
                continue
            published, state = set(), 128
            for kind, producer in order:
                if kind == "write":
                    published.add(producer)
                else:
                    state += 1
                if state == 128 + count:
                    assert len(published) == count
            arrival_orders += 1
    # Remove publication-before-arrival: all increments can precede every payload write.
    state, published = 128, set()
    unsafe_completion = False
    for kind, producer in [("arrive", i) for i in range(3)] + [("write", i) for i in range(3)]:
        if kind == "arrive":
            state += 1
        else:
            published.add(producer)
        if state == 131 and not published:
            unsafe_completion = True
    assert unsafe_completion
    # The previous graph replay has16 arrivals and old payload; only an epoch-aware wait rejects it.
    previous_state, current_base, count = 128 + 16, 256, 16
    assert previous_state % 128 == count
    assert previous_state != current_base + count
    assert ((1 << 57) * 128) & MASK64 == 0  # Scope must exclude tag wrap.
    return {
        "count_epoch_domains": domains,
        "small_publication_orders": arrival_orders,
        "stale_untagged_completion_rejected": True,
        "tag_wrap_outside_scope": True,
    }


def check_task_maps() -> dict[str, object]:
    cases = 0
    for tokens, sms in itertools.product((1, 2, 63, 64, 65, 137, 1025), (1, 3, 8, 132)):
        normal = [
            cta + ticket * sms
            for cta in range(sms)
            for ticket in range((tokens + sms - 1) // sms)
            if cta + ticket * sms < tokens
        ]
        assert sorted(normal) == list(range(tokens))
        pairs = (tokens + 1) // 2
        for packs, partitions in ((4, 2), (12, 2), (28, 14)):
            tasks = pairs * partitions
            shifted = []
            for cta in range(sms):
                for ticket in range((tasks + 4 * sms - 1) // (4 * sms) * 4):
                    task = cta * 4 + ticket % 4 + ticket // 4 * (sms * 4)
                    if task < tasks:
                        shifted.append(task)
            assert sorted(shifted) == list(range(tasks))
            segments = [
                range(i * packs // partitions, (i + 1) * packs // partitions)
                for i in range(partitions)
            ]
            assert [i for segment in segments for i in segment] == list(range(packs))
        cases += 1
    return {
        "token_grid_domains": cases,
        "odd_token_tail": 137,
        "shifted_hidden_partition_domains": [[4, 2], [12, 2], [28, 14]],
    }


def main() -> None:
    root = Path(json.loads((STUDY / "inputs/sources.json").read_text())["tree"])
    scheduler = (root / "deep_gemm/include/deep_gemm/scheduler/mega_mhc.cuh").read_text()
    for marker in (
        "(epoch + 1) * kNumSplitBarriers * kNumMaxSplits",
        "ptx::ld_acq_gpu(split_barrier_ptr) - current_barrier_base >= kNumSplits",
        "ptx::red_async_inc_rel(state_ptr)",
        "num_ctas - 1",
    ):
        assert marker in scheduler
    host = (root / "csrc/apis/mega_mhc.hpp").read_text()
    assert "math::ceil_div(num_k_blocks, math::ceil_div(num_k_blocks, max_num_splits))" in host
    admitted = []
    for n in range(8, 129, 8):
        block_n = (n + 15) // 16 * 16
        swizzle = next(mode for mode in (128, 64, 32, 16) if block_n * 4 % mode == 0)
        if swizzle // 4 == block_n:
            admitted.append(n)
    assert admitted == [8, 16, 24, 32]
    result = {
        "outcome": "PASS",
        "split_policy": check_split_policy(),
        "prenorm_N_passing_device_swizzle_condition": admitted,
        "sm100_host_device_menu_counterexample": {
            "N": 40,
            "block_N": 48,
            "swizzle_bytes": 64,
            "device_condition": "64/4 != 48",
        },
        "epochs": check_epoch(),
        "task_maps": check_task_maps(),
        "boundary": "Scalar index/count ordering only; no device visibility or residency proof.",
    }
    (STUDY / "checks/mhc-schedule-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write("PASS: split policy, repeated-epoch counts and Normal/Shifted task maps.\n")


if __name__ == "__main__":
    main()
