"""Native GEMM grouping and source epilogue coordinates; no CUDA execution."""

from __future__ import annotations

import itertools
import json
import struct
import sys
from pathlib import Path

from attention_values import bf16
from scaled_partials import decoded_dot, promoted_dot

STUDY = Path(__file__).resolve().parents[1]
MASK = 0xFFFFFFFF
COEFFICIENTS = (0x5671D42B, 0x9995E499, 0xACE1B8A5, 0xE153538D)


def random_half(h: int, offset: int) -> int:
    h &= MASK
    h ^= h >> 23
    h = h * 0x7FEB352D & MASK
    h ^= h >> 16
    h = h * (0x846CA68B if offset < 2 else 0xD35A2D97) & MASK
    h ^= h >> 11
    return (h >> ((offset % 2) * 16)) & 0xFFFF


def check_quartets() -> dict[str, int]:
    def bits(row: int, col: int) -> int:
        return struct.unpack("<I", struct.pack("<f", (row * 67 + col - 31) / 8))[0]

    coords, checked = {}, 0
    for lane, value in itertools.product(range(32), range(8)):
        row = 2 * (lane % 4) + value % 2
        col = lane // 16 * 4 + (lane // 4) % 4 + (value // 2) * 8
        assert (row, col) not in coords
        coords[row, col] = lane, value
    assert set(coords) == set(itertools.product(range(8), range(32)))
    for row, col in coords:
        lane, value = coords[row, col]
        offset = (lane // 4) % 4
        group = col // 4
        # Source swap-AB sums partial terms through XOR4 then XOR8.
        source_sum = (
            sum(
                bits(row, (col // 4) * 4 + ((peer // 4) % 4)) * COEFFICIENTS[(peer // 4) % 4]
                for peer in (lane, lane ^ 4, lane ^ 8, lane ^ 12)
            )
            & MASK
        )
        quartet_index = 2 * (value // 2) + lane // 16
        assert quartet_index == group
        oracle_sum = sum(bits(row, group * 4 + j) * COEFFICIENTS[j] for j in range(4)) & MASK
        assert random_half(source_sum + quartet_index, offset) == random_half(
            oracle_sum + group, col % 4
        )
        checked += 1
    sf_rows = [2 * (lane % 4) + lane // 4 for lane in range(8)]
    assert sorted(sf_rows) == list(range(8))
    return {"swapped_fragment_coordinates": len(coords), "cross_layout_hash_checks": checked}


def check_scale_storage() -> dict[str, int]:
    cases, writes = 0, 0
    for batches, rows, n in itertools.product((1, 2, 3), (1, 3, 17), (128, 256, 384)):
        stride = (rows + 3) // 4 * 4
        addresses = set()
        for batch, row, col in itertools.product(range(batches), range(rows), range(0, n, 32)):
            sf_index = (batch * n + col) // 32
            source = ((sf_index // 4) * stride + row) * 4 + sf_index % 4
            # Independent packed word by flattened (batch,N128 tile), then row, then byte.
            oracle = ((batch * (n // 128) + col // 128) * stride + row) * 4 + (col % 128) // 32
            assert source == oracle and source not in addresses
            addresses.add(source)
        writes += len(addresses)
        cases += 1
    return {"batched_sf_domains": cases, "unique_live_scale_bytes": writes}


def check_rounding() -> dict[str, object]:
    raw, bound = 1.00390625, 1.002
    assert raw > bound and bf16(raw) == bf16(bound) == 1
    rounded_then_count = int(abs(bf16(raw)) > bf16(bound))
    assert rounded_then_count == 0 and int(abs(raw) > bound) == 1
    # Source TMA BF16 accumulation rounds the product before addition to prior D.
    previous = 0.00390625
    native = bf16(bf16(raw) + previous)
    fp32_combine = bf16(raw + previous)
    assert native != fp32_combine
    return {
        "bf16_domain_clamp_count": rounded_then_count,
        "raw_domain_count_counterexample": 1,
        "bf16_product_then_accumulate": native,
        "fp32_combine_then_cast": fp32_combine,
    }


def check_stages() -> int:
    domains = 0
    for stages, normal, kmajor in itertools.product(range(1, 33), (False, True), (False, True)):
        merge = stages // 8 if stages >= 8 and normal and kmajor else 1
        merged_stages, block_k = stages // merge, 64 * merge
        assert merged_stages * block_k <= stages * 64
        if merge > 1:
            assert merged_stages >= 8
        for k in (1, 64, 65, 128, 257):
            visited = [
                j for base in range(0, k, block_k) for j in range(base, min(k, base + block_k))
            ]
            assert visited == list(range(k))
        domains += 1
    return domains


def check_per_column_promotion() -> int:
    a = [i % 7 - 3 for i in range(256)]
    b = [i % 11 - 5 for i in range(256)]
    cases = 0
    for row, col in itertools.product(range(3), range(5)):
        sa = [row + 1, 2 * row + 4]
        sb = [col + 2, 3 * col + 1]
        assert promoted_dot(a, b, sa, sb, 128, 128) == decoded_dot(a, b, sa, sb, 128)
        cases += 1
    return cases


def main() -> None:
    root = Path(json.loads((STUDY / "inputs/sources.json").read_text())["tree"])
    ops = (root / "deep_gemm/include/deep_gemm/epilogue/operators.cuh").read_text()
    for marker in (
        "quartet_offset == 0 ? 0x5671d42bu",
        "xor_mask = 4; xor_mask <= 8",
        "math::cast_into_bf16x2",
        "(batch_idx * args.shape_n + group_n_idx)",
    ):
        assert marker in ops
    result = {
        "outcome": "PASS",
        "quartets": check_quartets(),
        "scale_storage": check_scale_storage(),
        "rounding": check_rounding(),
        "stage_merge_domains": check_stages(),
        "sm90_per_column_scale_outputs": check_per_column_promotion(),
        "boundary": (
            "Scalar index/hash/rounding witnesses only; no cvt.rs, FP8 instruc"
            "tion or memory-order execution."
        ),
    }
    (STUDY / "checks/native-epilogues-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: epilogue fragment/hash identities, SF bytes, round boundaries and merged stages.\n"
    )


if __name__ == "__main__":
    main()
