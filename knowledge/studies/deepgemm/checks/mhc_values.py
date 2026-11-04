"""Source-shaped value, lane-group and scale-layout witnesses for Mega MHC."""

from __future__ import annotations

import json
import math
import struct
import sys
from fractions import Fraction
from pathlib import Path

from attention_values import bf16

STUDY = Path(__file__).resolve().parents[1]


def matrix_policy(logits: list[list[float]], epsilon: float, iterations: int) -> list[list[float]]:
    matrix = []
    for row in logits:
        maximum = max(row)
        exponentials = [math.exp(value - maximum) for value in row]
        denominator = sum(exponentials)
        matrix.append([value / denominator + epsilon for value in exponentials])
    for iteration in range(iterations):
        if iteration:
            matrix = [[value / (sum(row) + epsilon) for value in row] for row in matrix]
        columns = [sum(matrix[row][col] for row in range(4)) + epsilon for col in range(4)]
        matrix = [[matrix[row][col] / columns[col] for col in range(4)] for row in range(4)]
    return matrix


def lane_policy(
    logits: list[list[float]], epsilon: float, iterations: int, *, second_column_xor: int = 24
) -> list[list[float]]:
    values = [0.0] * 32
    for row in range(4):
        values[8 + row * 4 : 12 + row * 4] = logits[row]
    exponentials = [0.0] * 32
    for lane in range(32):
        start = lane // 4 * 4
        exponentials[lane] = math.exp(values[lane] - max(values[start : start + 4]))
    values = [
        exponentials[lane] / sum(exponentials[lane // 4 * 4 : lane // 4 * 4 + 4]) + epsilon
        for lane in range(32)
    ]
    for iteration in range(iterations):
        if iteration:
            sums = [sum(values[lane // 4 * 4 : lane // 4 * 4 + 4]) + epsilon for lane in range(32)]
            values = [value / sums[lane] for lane, value in enumerate(values)]
        partial = [values[lane] + values[lane ^ 4] for lane in range(32)]
        col_sum = [partial[lane] + partial[lane ^ second_column_xor] for lane in range(32)]
        values = [value / (col_sum[lane] + epsilon) for lane, value in enumerate(values)]
    return [values[8 + row * 4 : 12 + row * 4] for row in range(4)]


def check_sinkhorn() -> dict[str, object]:
    logits = [[float(row * col - 2 * row + col) / 3 for col in range(4)] for row in range(4)]
    domains = 0
    for epsilon in (0.0, 1e-3, 0.25):
        for iterations in (1, 2, 10):
            expected = matrix_policy(logits, epsilon, iterations)
            actual = lane_policy(logits, epsilon, iterations)
            assert (
                max(
                    abs(expected[row][col] - actual[row][col])
                    for row in range(4)
                    for col in range(4)
                )
                < 1e-12
            )
            domains += 1
    correct = matrix_policy(logits, 0.25, 2)
    wrong_lane = lane_policy(logits, 0.25, 2, second_column_xor=8)
    assert any(
        abs(correct[row][col] - wrong_lane[row][col]) > 1e-6 for row in range(4) for col in range(4)
    )
    assert matrix_policy(logits, 0.25, 1) != correct
    assert matrix_policy(logits, 0.0, 2) != correct
    column_sums = [sum(correct[row][col] for row in range(4)) for col in range(4)]
    assert all(value < 1.0 for value in column_sums)
    return {
        "domains": domains,
        "active_lanes": [8, 23],
        "column_xors": [4, 24],
        "finite_epsilon_column_sums": column_sums,
    }


def check_rounding() -> dict[str, object]:
    # Distinct input/output route coefficients distinguish C[i,j] from C[j,i].
    x, residual, post = 2, (3, 5, 7, 11), (1, 2, 4, 8)
    comb = [[1 + row * 4 + col for col in range(4)] for row in range(4)]
    proper = [
        x * post[col] + sum(residual[row] * comb[row][col] for row in range(4)) for col in range(4)
    ]
    transposed = [
        x * post[col] + sum(residual[row] * comb[col][row] for row in range(4)) for col in range(4)
    ]
    assert proper != transposed
    raw = [257 / 256, 513 / 256, 769 / 256, 1025 / 256]
    rounded = [bf16(value) for value in raw]
    rounded_stat = sum(Fraction(value) ** 2 for value in rounded)
    raw_stat = sum(Fraction(value) ** 2 for value in raw)
    assert rounded == [1.0, 2.0, 3.0, 4.0] and rounded_stat == 30 and raw_stat != rounded_stat
    fn = (1, 3, 5, 7)
    assert sum(rounded[i] * fn[i] for i in range(4)) != sum(raw[i] * fn[i] for i in range(4))
    # The Pre-combined norm input is rounded again before its RMS statistic.
    pre = [1 / 256] * 4
    norm_input_raw = sum(rounded[i] * pre[i] for i in range(4))
    norm_input = bf16(norm_input_raw)
    assert norm_input == norm_input_raw  # This particular value is exactly BF16.
    norm_input_raw = 1 + 1 / 256
    norm_input = bf16(norm_input_raw)
    assert (
        norm_input != norm_input_raw and norm_input * norm_input != norm_input_raw * norm_input_raw
    )
    # Every normal BF16 significand fits the logical TF32 mantissa after exact FP32 widening.
    widened = 0
    for exponent in range(1, 255):
        for mantissa in range(128):
            bits = ((exponent << 7) | mantissa) << 16
            assert bits & ((1 << 13) - 1) == 0
            widened += 1
    return {
        "route_oriented_outputs": proper,
        "raw_post": raw,
        "rounded_post": rounded,
        "hc_sqr_sum_after_rounding": str(rounded_stat),
        "before_rounding": str(raw_stat),
        "normal_bf16_significands_widened": widened,
    }


def byte_perm(a: int, b: int, selector: int) -> int:
    merged = a | (b << 32)
    return sum(
        ((merged >> (((selector >> (4 * byte)) & 7) * 8)) & 255) << (8 * byte) for byte in range(4)
    )


def check_scale_layouts() -> dict[str, object]:
    tags = [120 + lane // 4 for lane in range(32)]
    paired = [byte_perm(tags[lane], tags[lane ^ 4], 0x1140) for lane in range(32)]
    words = [byte_perm(paired[lane], paired[lane ^ 8], 0x5410) for lane in range(32)]
    assert words[0] == 0x7B7A7978 and words[16] == 0x7F7E7D7C
    row_domains = 0
    for block in (1, 32, 64, 128, 224, 256):
        for tokens in (1, 17, 223, 224, 225, 512):
            pitch = (block + 127) // 128 * 128
            rows = []
            for token in range(tokens):
                idx = token % block
                rows.append(
                    token // block * pitch + (idx & ~127) + (idx & 31) * 4 + ((idx >> 5) & 3)
                )
            assert len(rows) == len(set(rows))
            assert max(rows) < (tokens + block - 1) // block * pitch
            row_domains += 1
    scales = 0
    for exponent in range(1, 255):
        for mantissa in range(128):
            amax_bits = exponent << 7 | mantissa
            code = max((amax_bits + 127 - 96) >> 7, 105 + 8) - 8
            amax = struct.unpack("<f", struct.pack("<I", amax_bits << 16))[0]
            scale = 2.0 ** (code - 127)
            assert scale * 448 >= amax
            if code > 105:
                assert scale / 2 * 448 < amax
            scales += 1
    return {
        "packed_sf_word_tags": [hex(words[0]), hex(words[16])],
        "shared_row_domains": row_domains,
        "bf16_amax_exponent_ceiling_cases": scales,
        "boundary": (
            "Scale range and word/layout equations; FP8 conversion itself is not emulated."
        ),
    }


def main() -> None:
    root = Path(json.loads((STUDY / "inputs/sources.json").read_text())["tree"])
    epilogue = (root / "deep_gemm/include/deep_gemm/epilogue/sm100_mega_mhc.cuh").read_text()
    body = (root / "deep_gemm/include/deep_gemm/impls/sm100_mega_mhc.cuh").read_text()
    for marker in (
        "value * math::fast_rcp(row_sum) + mix_args.sinkhorn_eps",
        "element, kNumRoutes",
        "sum, 6 * kNumRoutes",
        "get_shared_sf_row_idx",
    ):
        assert marker in epilogue
    assert "input_route_idx * kNumRoutes + output_route_idx" in body
    assert "post_value = __bfloat1622float2(post_value_bf16)" in body
    result = {
        "outcome": "PASS",
        "sinkhorn": check_sinkhorn(),
        "rounding": check_rounding(),
        "scale_layouts": check_scale_layouts(),
        "boundary": "Scalar value/index witnesses; no GPU fast-math, TMA or TF32 execution.",
    }
    (STUDY / "checks/mhc-values-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write("PASS: finite row/column policy, shared BF16 boundaries and scale layouts.\n")


if __name__ == "__main__":
    main()
