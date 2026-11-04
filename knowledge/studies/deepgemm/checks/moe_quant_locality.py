"Source-equation checks for quantized epilogues and locality queue assumptions."

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def balance(mapping: list[int]) -> list[int]:
    larger = int(sum(mapping) * 2 > len(mapping))
    chosen = [i for i, domain in enumerate(mapping) if domain == larger][: len(mapping) // 2]
    result = mapping[:]
    for i in chosen:
        result[i] = larger ^ 1
    return result


def check_balance() -> dict[str, object]:
    cases, balanced, changed = 0, 0, 0
    for tpcs in (2, 4, 6, 8):
        for values in itertools.product((0, 1), repeat=tpcs):
            result = balance(list(values))
            expected_ones = sum(values) + (tpcs // 2 if sum(values) * 2 <= tpcs else -tpcs // 2)
            assert sum(result) == expected_ones
            cases += 1
            balanced += sum(result) * 2 == tpcs
            changed += sum(values) * 2 == tpcs and sum(result) * 2 != tpcs
    witness = [0, 1, 0, 1]
    assert balance(witness) == [1, 1, 1, 1]
    # With one wave the source disables work stealing; domain0's tasks then have no claimant.
    sms, tasks_per_domain = 8, 2
    total_tasks = tasks_per_domain * 2
    assert total_tasks < 2 * (sms // 2)
    claimed_domains = set(balance(witness))
    assert 0 not in claimed_domains
    return {
        "mapping_domains": cases,
        "balanced_outputs": balanced,
        "initially_balanced_inputs_broken": changed,
        "witness": {"before_tpc_domains": witness, "after_tpc_domains": balance(witness)},
        "unclaimed_queue_witness": {
            "sms": sms,
            "tasks_per_domain": tasks_per_domain,
            "stealing_enabled": False,
            "unclaimed_domain": 0,
        },
    }


def check_amax_groups() -> dict[str, int]:
    groups = 0
    for warp in range(4):
        # Two values per lane per token; identifiable coordinates reveal wrong XOR directions.
        values = [[1000 * warp + 10 * lane + feature for feature in range(2)] for lane in range(32)]
        reduced = [max(v) for v in values]
        for offset in (4, 8, 16):
            prior = reduced[:]
            reduced = [max(prior[lane], prior[lane ^ offset]) for lane in range(32)]
        peer = warp ^ 1
        for residue in range(4):
            lanes = [lane for lane in range(32) if lane % 4 == residue]
            local_oracle = max(values[lane][feature] for lane in lanes for feature in range(2))
            assert all(reduced[lane] == local_oracle for lane in lanes)
            peer_oracle = max(
                1000 * peer + 10 * lane + feature for lane in lanes for feature in range(2)
            )
            assert max(local_oracle, peer_oracle) == max(
                1000 * w + 10 * lane + feature
                for w in (warp, peer)
                for lane in lanes
                for feature in range(2)
            )
            assert len(lanes) * 2 * 2 == 32
            groups += 1
    return {"warp_pair_row_groups": groups, "features_per_scale": 32}


def check_sf_writes() -> dict[str, int]:
    domains, writes = 0, 0
    for block_m, store_m in ((16, 8), (32, 16), (64, 16), (128, 32), (192, 32), (240, 24)):
        sf_block = (block_m + 127) // 128 * 128
        stride = sf_block * 3
        addresses = set()
        expected = set()

        def transform(row: int) -> int:
            return (row & ~127) + (row & 31) * 4 + ((row >> 5) & 3)

        for row, kcode in itertools.product(range(block_m), range(8)):
            expected.add((kcode // 4) * stride * 4 + transform(row) * 4 + kcode % 4)
        for wg, store, atom, warp, lane, nblock in itertools.product(
            range(2),
            range((block_m // 2) // store_m),
            range(store_m // 8),
            (0, 2),
            range(4),
            range(4),
        ):
            row_base = wg * (block_m // 2) + store * store_m + atom * 8
            code = nblock * 2 + warp // 2
            address = (code // 4) * stride * 4 + (transform(row_base) + lane * 8) * 4 + code % 4
            for offset in (0, 16):
                assert address + offset not in addresses
                addresses.add(address + offset)
        assert addresses == expected
        writes += len(addresses)
        domains += 1
    return {"tile_domains": domains, "distinct_scale_byte_writes": writes}


def check_prefetch() -> dict[str, int]:
    cases = 0
    for block_m, store_m in (
        (16, 8),
        (32, 16),
        (64, 16),
        (128, 32),
        (192, 32),
        (240, 24),
        (240, 8),
    ):
        rows = block_m // 2
        total = rows // store_m
        prefetch = min(int(block_m == 240), total - 1)
        for valid in range(rows + 1):
            blocks = (valid + store_m - 1) // store_m
            loaded, consumed, released = [], [], []
            for s in range(prefetch):
                if s < blocks:
                    loaded.append(s)
                    if s + 1 == blocks:
                        released.append(len(loaded))
            if blocks == 0:
                released.append(0)
            for s in range(blocks):
                target = s if prefetch == 0 else s + prefetch
                if target < blocks:
                    loaded.append(target)
                    if target + 1 == blocks:
                        released.append(len(loaded))
                assert s in loaded
                consumed.append(s)
            assert loaded == consumed == list(range(blocks))
            assert released == [blocks]
            cases += 1
    return {"valid_row_domains": cases}


def main() -> None:
    root = Path(json.loads((STUDY / "inputs/sources.json").read_text())["tree"])
    host = (root / "csrc/apis/locality_domain.hpp").read_text()
    assert ".slice(0, num_tpcs / 2)" in host
    device = (root / "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh").read_text()
    for marker in (
        "math::warp_reduce<4, true>",
        "epilogue_warp_idx ^ 1",
        "sf_base_ptr[sf_addr + 4 * static_cast<uint32_t>(sizeof(uint32_t))]",
        "kNumEpiPrefetchStages = BLOCK_M == 240 ? 1 : 0",
    ):
        assert marker in device
    result = {
        "outcome": "PASS",
        "locality": check_balance(),
        "amax": check_amax_groups(),
        "sf_writes": check_sf_writes(),
        "prefetch": check_prefetch(),
        "boundary": (
            "Source arithmetic and event order only; no GPU probe or quantized"
            " instruction execution."
        ),
    }
    (STUDY / "checks/moe-quant-locality-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    sys.stdout.write(
        "PASS: scale-group ownership, SF bytes, prefetch lifetime and locality counterexample.\n"
    )


if __name__ == "__main__":
    main()
