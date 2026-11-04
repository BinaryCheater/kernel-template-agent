"Scalar witnesses for MQA value ordering, compressed outputs and paged addresses."

from __future__ import annotations

import itertools
import json
import struct
import sys
from fractions import Fraction
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def bf16(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def paired_reduction(scores: list[float], weights: list[float]) -> float:
    # Witness operands keep products/sums exactly representable in FP32 before BF16 rounding.
    partials = [0.0] * 4
    for base in range(0, len(scores), 4):
        for lane in range(4):
            term = bf16(max(scores[base + lane], 0.0)) * weights[base + lane]
            partials[lane] = bf16(term + partials[lane])
    pair = [bf16(partials[0] + partials[2]), bf16(partials[1] + partials[3])]
    return bf16(pair[0] + pair[1])


def main() -> None:
    root = Path(json.loads((STUDY / "inputs/sources.json").read_text())["tree"])
    core = (root / "deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh").read_text()
    api = (root / "csrc/apis/attention.hpp").read_text()
    sm90 = (root / "deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh").read_text()
    for marker in (
        "ptx::cvt_relu_bf16x2_f32",
        "__hfma2",
        "__hadd2_rn",
        "- seq_k_start[i]",
        "seq_k_start[i] <= kv_offset",
    ):
        assert marker in core
    assert "(sum[0] + sum[1]) * scale_kv_0" in sm90
    assert "fused_kv_cache.data_ptr<uint8_t>() + block_kv * kv_head_dim" in api
    homogeneous_cases = 0
    for scale in (Fraction(0), Fraction(1, 4), Fraction(1), Fraction(8)):
        for scores in itertools.product(
            (Fraction(-3), Fraction(0), Fraction(1, 2), Fraction(4)), repeat=3
        ):
            weights = (Fraction(1), Fraction(-2), Fraction(3))
            original = sum(w * max(scale * x, 0) for w, x in zip(weights, scores, strict=True))
            factored = scale * sum(w * max(x, 0) for w, x in zip(weights, scores, strict=True))
            assert original == factored
            homogeneous_cases += 1
    assert max(-2 * -1, 0) != -2 * max(-1, 0)
    assert sum((1, 3)) != 3 * sum((1, 1))  # A scale changing between terms cannot be hoisted.
    assert bf16(1 + 2.0**-8) == 1.0
    assert bf16(1 + 3 * 2.0**-8) == 1 + 2.0**-6
    scores, weights = [1.0] * 8, [256.0, 1.0, -256.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    reduced = paired_reduction(scores, weights)
    exact = sum(scores[i] * weights[i] for i in range(8))
    assert reduced == 5.0 and exact == 6.0
    # Head-local ReLU and signed weight must not be replaced by ReLU after head summation.
    assert max(1, 0) * 1 + max(-2, 0) * 1 != max(1 - 2, 0)
    assert max(1, 0) * -1 != max(1 * -1, 0)
    # Q-block span is a common aligned cover; each row stores only its valid compressed columns.
    starts, ends, split_kv = (5, 2, 8), (9, 7, 8), 384
    base = min(starts) // 4 * 4
    splits = (max(ends) - base + split_kv - 1) // split_kv
    written = {
        q: [(k, k - start) for k in range(base, base + splits * split_kv) if start <= k < ends[q]]
        for q, start in enumerate(starts)
    }
    assert written[0] == [(5, 0), (6, 1), (7, 2), (8, 3)]
    assert written[1] == [(2, 0), (3, 1), (4, 2), (5, 3), (6, 4)]
    assert written[2] == []
    # Retained fused cache is structure-of-arrays within each page, despite its 4D shape.
    page_kv, token_bytes, page_stride = 64, 128, 64 * (128 + 4)
    pages = (3, 1, 4)
    addresses = []
    for logical in (0, 63, 64, 66):
        page_index, token = divmod(logical, page_kv)
        physical = pages[page_index]
        kv_addr = physical * page_stride + token * token_bytes
        sf_addr = physical * page_stride + page_kv * token_bytes + token * 4
        wrong_interleaved = physical * page_stride + token * (token_bytes + 4)
        if token:
            assert kv_addr != wrong_interleaved
        addresses.append(
            {
                "logical_token": logical,
                "physical_page": physical,
                "kv_byte_address": kv_addr,
                "sf_byte_address": sf_addr,
            }
        )
    # Sparse-page metadata resolves logical sparse block -> page -> in-page block.
    blocks_per_page, sparse_extent = page_kv // 8, 8
    for logical_block in (0, 7, 8, 9):
        page_index, in_page_block = divmod(logical_block, blocks_per_page)
        physical_block = pages[page_index] * blocks_per_page + in_page_block
        physical_page, offset = divmod(physical_block, blocks_per_page)
        assert physical_page == pages[page_index]
        assert offset * sparse_extent == (logical_block * sparse_extent) % page_kv
    # Paged output stride holds a common multiple of split384 and BF16 row alignment512.
    alignment = 384 * 512 // 128
    assert alignment == 1536
    assert ((513 + alignment - 1) // alignment) * alignment == 1536
    result = {
        "outcome": "PASS",
        "exact_nonnegative_hoists": homogeneous_cases,
        "bf16_head_reduction_witness": {
            "scores": scores,
            "weights": weights,
            "source_shaped_result": reduced,
            "exact_result": exact,
        },
        "compressed_written_columns": {
            str(q): [column for _, column in items] for q, items in written.items()
        },
        "paged_addresses": addresses,
        "paged_stride_elements": 1536,
        "mutations_rejected": [
            "negative_scale_hoist",
            "per_head_scale_hoist",
            "relu_after_head_reduce",
            "weight_inside_relu",
            "token_interleaved_cache",
        ],
        "boundary": (
            "Bounded scalar arithmetic/address witnesses; no GPU numerical com"
            "parison or FP8 MMA emulation."
        ),
    }
    (STUDY / "checks/attention-values-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        f"PASS: {homogeneous_cases} exact hoists, BF16 rounding "
        "and compressed/page address witnesses.\n"
    )


if __name__ == "__main__":
    main()
