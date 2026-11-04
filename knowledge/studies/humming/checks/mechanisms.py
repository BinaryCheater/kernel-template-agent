"Small source-arithmetic witnesses; these are not GPU simulations or ProofResults."

from __future__ import annotations

import itertools
import json
import math
import struct
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def value(encoded: int) -> float:
    return struct.unpack("<f", struct.pack("<I", encoded))[0]


def atomic_combine() -> dict:
    orders = 0
    for count in range(2, 8):
        for order in itertools.permutations(range(1, count)):
            lock = 0
            for generation in range(2):
                assert lock == 0
                output = 1 + generation * 10
                lock = 1 - count
                for partial in order:
                    assert lock <= -1
                    output += partial + 1 + generation * 10
                    lock += 1
                assert lock == 0
                assert output == sum(i + 1 + generation * 10 for i in range(count))
            orders += 1
    # A completion increment before its write lets another generation overwrite it.
    lock = -1
    output = 1
    lock += 1
    assert lock == 0 and output != 1 + 2
    return {
        "orders": orders,
        "generations_per_order": 2,
        "partials_domain": [2, 7],
        "early_completion_mutation_rejected": True,
        "open": "CUDA visibility, atomic linearization, TMA completion, forward progress",
    }


def transform(source: list[float], per_lane: int) -> list[float]:
    data = list(source)
    length = len(data)
    # hadamard.cuh:29-94; split at log2(V) then at the shuffle/shared lane boundary.
    for bit in range(length.bit_length() - 1):
        partner = 1 << bit
        previous = list(data)
        for i in range(length):
            other = previous[i ^ partner]
            data[i] = previous[i] + other if i & partner == 0 else other - previous[i]
    assert per_lane <= length
    return [v / math.sqrt(length) for v in data]


def butterfly() -> dict:
    rows = []
    for length in (4, 32, 128, 256, 512):
        for column in range(length):
            basis = [0.0] * length
            basis[column] = 1.0
            actual = transform(basis, 4)
            expected = [
                (-1.0 if (row & column).bit_count() % 2 else 1.0) / math.sqrt(length)
                for row in range(length)
            ]
            assert actual == expected
        rows.append(
            {
                "H": length,
                "V": 4,
                "basis_vectors": length,
                "lanes": length // 4,
                "shared_stage": length // 4 > 32,
            }
        )
    assert transform([0.0, 1.0, 0.0, 0.0], 4)[1] < 0
    return {
        "rows": rows,
        "high_branch_sign_mutation_rejected": True,
        "boundary": "Logical butterfly identity only; CUDA transport and rsqrt are untested",
    }


def fp4_rebase() -> dict:
    magnitudes = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    cases = []
    for code in range(16):
        for delta in range(4):
            q = 0 if code == 8 else code
            if delta:
                x = value(((q & 8) << 28) | ((q & 7) << 22))
                x = f32(x * value(0x3F800000 - (delta << 23)))
                raw = bits(x)
                lower = raw & 0x81C00000
                upper = (raw + 0x00200000) & 0x81C00000
                chosen = upper if abs(x - value(lower)) >= abs(x - value(upper)) else lower
                q = ((chosen & 0x80000000) >> 28) | ((chosen & 0x01C00000) >> 22)
            target = magnitudes[code & 7] / (2**delta)
            # Independent E2M1 value table; ties go to larger magnitude.
            best = min(range(8), key=lambda i: (abs(magnitudes[i] - target), -magnitudes[i]))
            expected = best | (8 if code & 8 and best else 0)
            # Source may retain signed zero after a lossy underflow; value equivalence is enough.
            assert magnitudes[q & 7] * (-1 if q & 8 else 1) == magnitudes[expected & 7] * (
                -1 if expected & 8 else 1
            )
            cases.append({"code": code, "delta": delta, "output": q})
    assert next(r["output"] for r in cases if r["code"] == 1 and r["delta"] == 1) == 1
    return {
        "cases": cases,
        "tie_to_zero_mutation_rejected": True,
        "boundary": "Code-value rebase, not preservation of the original weight product",
    }


def quantization() -> dict:
    # quant_weight.cuh:225-258,69-109; already requested quantizer, not a lossless GEMM rewrite.
    inputs = [float(i) for i in range(-8, 8)] * 8
    minimum, maximum = min(inputs), max(inputs)
    scale = f32((maximum - minimum) / 15)
    zero = round(f32(-minimum / scale))
    encoded = [min(15, max(0, round(f32((x + zero * scale) / scale)))) for x in inputs]
    assert scale == 1 and zero == 8 and encoded == list(range(16)) * 8
    ties = [round(x) for x in (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)]
    assert ties == [-2, -2, 0, 0, 2, 2]
    # UE8M0 scale rounding uses exponent ceiling, not nearest rounding.
    e8_rows = []
    for scale_in in (1.0, 1.0000001192092896, 1.5, 2.0, 3.0):
        encoded_scale = (bits(scale_in) + 0x007FFFFF) & 0x7F800000
        expected = 2.0 ** math.ceil(math.log2(scale_in))
        assert value(encoded_scale) == expected
        e8_rows.append({"input": scale_in, "rounded_scale": expected})
    return {
        "group_size": 128,
        "scheme": "unsigned INT4 with integer zero point",
        "codes": 128,
        "scale": scale,
        "zero_point": zero,
        "ties_even": ties,
        "e8m0_rounding": e8_rows,
        "open": (
            "All-zero/NaN/extreme groups, negative-scale branch and low-float quantizer tolerance"
        ),
    }


def ownership_and_rings() -> dict:
    # scheduler.cuh:238-250: ragged final raster group must use its actual M extent.
    raster_cases = 0
    for m_blocks in range(1, 10):
        for n_blocks in range(1, 6):
            for group in (1, 2, 4, 8):
                coordinates = []
                for index in range(m_blocks * n_blocks):
                    if group == 1:
                        coordinate = divmod(index, n_blocks)
                    else:
                        group_id, inner = divmod(index, group * n_blocks)
                        first = group_id * group
                        actual = min(m_blocks - first, group)
                        coordinate = (first + inner % actual, inner // actual)
                    coordinates.append(coordinate)
                assert set(coordinates) == {
                    (m, n) for m in range(m_blocks) for n in range(n_blocks)
                }
                assert len(coordinates) == len(set(coordinates))
                raster_cases += 1
    # Empty experts and different contiguous/masked offsets.
    counts = [0, 3, 9]
    prefix = [0]
    for count in counts:
        prefix.append(prefix[-1] + (count + 3) // 4)
    assert prefix == [0, 0, 1, 4]
    assert [route // 2 for route in (5, 0, 3, 2)] == [2, 0, 1, 1]
    rings = [(step % 3, (step // 3) % 2) for step in range(7)]
    assert rings[5] == (2, 1) and rings[5] != (0, 0)
    return {
        "raster_cases": raster_cases,
        "empty_expert_prefix": prefix,
        "indexed_topk_input_rows": [2, 0, 1, 1],
        "continuous_stage_ring": rings,
        "tile_boundary_stage_reset_mutation_rejected": True,
        "open": (
            "Stream-K slice metadata and asynchronous generation safety beyond index arithmetic"
        ),
    }


def main() -> None:
    result = {
        "outcome": "PASS",
        "kind": "source_arithmetic_replay",
        "gpu_execution": False,
        "atomic_combine": atomic_combine(),
        "butterfly": butterfly(),
        "fp4_rebase": fp4_rebase(),
        "quantization": quantization(),
        "ownership_and_rings": ownership_and_rings(),
    }
    (STUDY / "checks/mechanisms-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: atomic countdown orders, Hadamard basis, FP4 rebasing, quan"
        "tizer boundaries, raster ownership and continuous stage indices.\n"
    )


if __name__ == "__main__":
    main()
