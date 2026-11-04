"""Exact-rational ownership witnesses; this does not execute a CUDA kernel."""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def decoded_dot(a: list[int], b: list[int], sa: list[int], sb: list[int], grain: int) -> Fraction:
    return sum(
        (Fraction(x * sa[k // grain]) * Fraction(b[k] * sb[k // grain]) for k, x in enumerate(a)),
        Fraction(0),
    )


def promoted_dot(
    a: list[int],
    b: list[int],
    sa: list[int],
    sb: list[int],
    grain: int,
    partial: int,
    *,
    stale: bool = False,
) -> Fraction:
    assert grain % partial == 0
    final, raw = Fraction(0), Fraction(0)
    for start in range(0, len(a), partial):
        if not stale:
            raw = Fraction(0)
        raw += sum(
            (Fraction(a[k] * b[k]) for k in range(start, min(start + partial, len(a)))), Fraction(0)
        )
        final += raw * sa[start // grain] * sb[start // grain]
    return final


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    body = (
        Path(manifest["tree"]) / "deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh"
    ).read_text()
    for marker in (
        "BLOCK_K == 128",
        "scale_a_0 * scale_b_0",
        "scale_a_0 * scale_b_1",
        "i < num_former_iters",
        "WGMMA::wgmma(a_desc, b_desc, accum, k)",
        ("shifted_accum[i * 4 + 0] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 0]"),
    ):
        assert marker in body
    # N tile [96,192) crosses the per-128 B scale boundary at column 128.
    # Row and column tags make swapped scale owners distinguishable.
    grain, length = 128, 256
    a = [((k * 7) % 9) - 4 for k in range(length)]
    b = [((k * 3) % 11) - 5 for k in range(length)]
    sa_rows = [[1, 8], [2, 16]]
    sb_columns = [[4, 2], [32, 1]]
    witnesses = []
    for m in (0, 1):
        for n in (96, 127, 128, 191):
            sa, sb = sa_rows[m], sb_columns[n // 128]
            expected = decoded_dot(a, b, sa, sb, grain)
            observed = promoted_dot(a, b, sa, sb, grain, grain)
            assert observed == expected
            wrong_final = sum(a[k] * b[k] for k in range(length)) * sa[-1] * sb[-1]
            wrong_owner = promoted_dot(a, b, sa, sb_columns[1 - n // 128], grain, grain)
            assert wrong_final != expected
            assert wrong_owner != expected
            assert promoted_dot(a, b, sa, sb, grain, grain, stale=True) != expected
            witnesses.append({"m": m, "n": n, "exact_value": str(expected)})
    # Algebraic divisors and a short final K partial exercise the general constraint.
    tails = 0
    for length in (1, 127, 128, 129, 255, 256):
        a, b = list(range(1, length + 1)), [1] * length
        for partial in (1, 2, 4, 16, 32, 64, 128):
            assert promoted_dot(a, b, [2, 8], [4, 1], grain, partial) == decoded_dot(
                a, b, [2, 8], [4, 1], grain
            )
            tails += 1
    try:
        promoted_dot([1] * 256, [1] * 256, [1, 3], [1, 1], grain, 256)
    except AssertionError:
        invalid_rejected = True
    else:
        raise AssertionError("A scale-crossing partial was accepted")
    result = {
        "outcome": "PASS",
        "sm90_source_bindings": {
            "partial_k": 128,
            "gran_k_a": 128,
            "gran_k_b": 128,
            "gran_m": 1,
            "gran_n": 128,
            "tile_n": 96,
            "n_block": 1,
        },
        "witnesses": witnesses,
        "algebraic_tail_divisor_cases": tails,
        "mutations_rejected": [
            "single_final_scale",
            "wrong_B_scale_owner",
            "stale_raw_accumulator",
        ],
        "scale_crossing_partial_rejected": invalid_rejected,
        "boundary": (
            "Exact rational algebra and source markers; no CUDA execution or f"
            "loating-point error bound."
        ),
    }
    (STUDY / "checks/scaled-partials-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        f"PASS: {len(witnesses)} scale-owner witnesses; {tails} tail/divisor cases; "
        "four wrong explanations rejected."
    )


if __name__ == "__main__":
    main()
    sys.stdout.write("\n")
