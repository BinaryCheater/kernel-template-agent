"""MegaMoE layout, rank interleave and parity equations; no device execution."""

from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path

from attention_values import bf16

STUDY = Path(__file__).resolve().parents[1]
BLOCKS = (8, 16, 32, 64, 96, 128, 192, 240)


def align(n: int, width: int) -> int:
    return (n + width - 1) // width * width


def check_weights() -> dict[str, int]:
    cases, coordinates = 0, 0
    for groups, rows, packed_k in itertools.product((1, 3), (256, 512), (1, 3, 7)):
        half = rows // 2
        # Literal stack([gate,up],dim=2).reshape row order.
        interleaved = [
            side * half + block * 8 + j
            for block in range(half // 8)
            for side in range(2)
            for j in range(8)
        ]
        # Literal reshape [rows/128,4,32] then transpose the inner two axes.
        transpose = [
            block * 128 + outer * 32 + inner
            for block in range(rows // 128)
            for inner in range(32)
            for outer in range(4)
        ]
        assert sorted(interleaved) == sorted(transpose) == list(range(rows))
        actual = [interleaved[index] for index in transpose]
        for group, row, k in itertools.product(range(groups), range(rows), range(packed_k)):
            prior_row = row // 128 * 128 + (row % 4) * 32 + (row % 128) // 4
            expected_original = (prior_row % 16 // 8) * half + prior_row // 16 * 8 + prior_row % 8
            assert (group, actual[row], k) == (group, expected_original, k)
            coordinates += 1
        assert actual != [transpose[index] for index in interleaved]
        cases += 1
    return {"shape_domains": cases, "tagged_coordinates": coordinates}


def peel(counts: list[int], index: int) -> tuple[int, int]:
    remaining, offset = counts[:], 0
    while True:
        active = [rank for rank, count in enumerate(remaining) if count > 0]
        length = min(remaining[rank] for rank in active)
        if index < length * len(active):
            return active[index % len(active)], offset + index // len(active)
        index -= length * len(active)
        offset += length
        remaining = [max(0, count - length) for count in remaining]


def check_rank_interleave() -> dict[str, object]:
    rng, cases = random.Random(410), 0
    counts_cases = [list(c) for c in itertools.product(range(4), repeat=4)]
    counts_cases += [[rng.randrange(8) for _ in range(ranks)] for ranks in (1, 31, 32, 33, 64, 72)]
    for counts in counts_cases:
        oracle = [
            (rank, slot)
            for slot in range(max(counts))
            for rank, count in enumerate(counts)
            if slot < count
        ]
        assert [peel(counts, i) for i in range(sum(counts))] == oracle
        assert len(set(oracle)) == sum(counts)
        cases += 1
    return {
        "count_domains": cases,
        "witness_counts": [3, 0, 1, 2],
        "source_order": [peel([3, 0, 1, 2], i) for i in range(6)],
    }


def check_sf_ring() -> dict[str, object]:
    cases = 0
    ring_tokens = 1920
    capacity = max(ring_tokens // block * align(block, 128) for block in BLOCKS)
    for block in BLOCKS:
        sf_block = align(block, 128)
        addresses = []
        for logical_block, token in itertools.product(
            range(ring_tokens // block * 2), range(block)
        ):
            ring_block = logical_block % (ring_tokens // block)
            source = ring_block * sf_block + (token & ~127) + (token & 31) * 4 + ((token >> 5) & 3)
            oracle = ring_block * sf_block + token // 128 * 128 + token % 32 * 4 + token % 128 // 32
            assert source == oracle and source < capacity
            if logical_block < ring_tokens // block:
                addresses.append(source)
        assert len(set(addresses)) == ring_tokens
        cases += 1
    assert capacity == 30720  # M8 worst case, sixteen SF rows per payload token.
    # SF scratch and full source metadata have different lifetimes and capacities.
    old = {"rank": 0, "token": 4, "slot": 1}
    newer = {"rank": 1, "token": 7, "slot": 0}
    full_metadata = {0: old, ring_tokens: newer}
    wrong_ring_metadata = {0: old}
    wrong_ring_metadata[ring_tokens % ring_tokens] = newer
    assert full_metadata[0] != wrong_ring_metadata[0]
    return {
        "block_domains": cases,
        "ring_tokens": ring_tokens,
        "worst_case_sf_rows": capacity,
        "metadata_alias_counterexample": True,
    }


def check_address_ownership() -> dict[str, object]:
    bases = [0x100000, 0x900000, 0x3100000]
    mappings = 0
    for local_rank, remote_rank, offset in itertools.product(
        range(3), range(3), (0, 16, 1024, 65520)
    ):
        local_pointer = bases[local_rank] + offset
        mapped = local_pointer + bases[remote_rank] - bases[local_rank]
        assert mapped == bases[remote_rank] + offset
        mappings += 1
    capacity, ranks = 1920, 2

    def source_index(expert: int, rank: int, slot: int) -> int:
        return (expert * ranks + rank) * capacity + slot

    # Duplicate expert choices can exceed a per-expert/source-rank table row.
    assert source_index(0, 0, capacity) == source_index(0, 1, 0)
    assert source_index(0, 0, ranks * capacity) == source_index(1, 0, 0)
    # The last partial block's contribution reaches full readiness without claiming more valid rows.
    valid, width = 19, 16
    contributions = [1] * (valid - 1) + [width - (valid - 1) % width]
    assert sum(contributions) == align(valid, width) == 32
    return {
        "symmetric_pointer_mappings": mappings,
        "duplicate_expert_source_row_collision": True,
        "partial_tile_valid_rows": valid,
        "published_readiness_units": sum(contributions),
    }


def check_parity() -> dict[str, object]:
    orders = 0
    for fragments in range(1, 7):
        full, mask = (1 << fragments) - 1, 0
        for order in itertools.permutations(range(fragments)):
            for generation in range(4):
                expected = full if generation % 2 == 0 else 0
                completed = set()
                for fragment in order:
                    mask ^= 1 << fragment
                    completed.add(fragment)
                    for size in range(1, fragments + 1):
                        if fragments % size:
                            continue
                        for start in range(0, fragments, size):
                            group = ((1 << size) - 1) << start
                            ready = (mask ^ expected) & group == 0
                            assert ready == set(range(start, start + size)).issubset(completed)
                assert mask == expected
                orders += 1
    # Duplicate toggles can lose a publication. Two later toggles can alias old parity.
    assert 0 ^ 1 ^ 1 == 0
    stale_expected, mask = 1, 1
    mask ^= 1  # Generation1 publishes before an outstanding generation0 read.
    mask ^= 1  # Generation2 now aliases generation0 while holding different payload.
    assert mask == stale_expected
    # One required fragment suffices even while another fragment is absent.
    assert ((1 ^ 3) & 1) == 0 and (1 ^ 3) != 0
    # Explicit full-width mask avoids the source's special-cased shift by64.
    full64 = (1 << 64) - 1
    assert full64.bit_count() == 64
    return {
        "publication_orders_across_generations": orders,
        "duplicate_toggle_rejected_by_premise": True,
        "generation_lap_alias": True,
        "partial_read_before_full_mask": True,
    }


def check_numerical_boundary() -> dict[str, object]:
    # Quantization prevents moving a routed scalar across the L2 input round.
    value, weight = 1.0, 1.00390625
    before = bf16(value * weight)
    after = bf16(value) * weight
    assert before == 1.0 and before != after
    # Gate is upper-clamped only; up is symmetrically clamped.
    gate, up, clamp = -20.0, -20.0, 10.0
    assert min(bf16(gate), clamp) == -20.0
    assert max(-clamp, min(bf16(up), clamp)) == -10.0
    return {
        "weighted_round": before,
        "incorrect_final_only_weight": after,
        "asymmetric_clamp": {"gate": -20, "up": -10},
    }


def main() -> None:
    root = Path(json.loads((STUDY / "inputs/sources.json").read_text())["tree"])
    scheduler = (root / "deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh").read_text()
    for marker in (
        "ptx::red_xor_rel",
        "pending_mask & k_block_mask",
        "generation_idx & 1",
        "task_info_empty_barriers[sched_stage_idx].wait(sched_phase ^ 1)",
    ):
        assert marker in scheduler
    layout = (root / "deep_gemm/include/deep_gemm/layout/mega_moe.cuh").read_text()
    assert "{8, 16, 32, 64, 96, 128, 192, 240}" in layout
    result = {
        "outcome": "PASS",
        "weights": check_weights(),
        "rank_interleave": check_rank_interleave(),
        "sf_ring": check_sf_ring(),
        "parity": check_parity(),
        "address_ownership": check_address_ownership(),
        "numeric_boundary": check_numerical_boundary(),
        "boundary": (
            "Scalar source equations only; memory visibility, liveness, ISA an"
            "d quantization instructions remain open."
        ),
    }
    (STUDY / "checks/moe-layout-schedule-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    sys.stdout.write(
        "PASS: weight/SF layout, rank interleave, ring parity and rounding boundaries.\n"
    )


if __name__ == "__main__":
    main()
