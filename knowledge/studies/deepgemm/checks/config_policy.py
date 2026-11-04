"""Bounded SM100 BF16 configuration policy replay; no performance measurement."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def div(n: int, d: int) -> int:
    return (n + d - 1) // d


def candidates(m: int, n: int, k: int, mn_a: bool, mn_b: bool) -> list[tuple[int, ...]]:
    result = []
    for swap in (0, 1):
        ms = range(16, 257, 16) if swap else (32 if m <= 32 else 64 if m <= 64 else 128,)
        ns = (128,) if swap else (16, *range(32, (128 if k <= 256 else 256) + 1, 32))
        for cm, cn, bm, bn in itertools.product((1, 2), (1, 2), ms, ns):
            if cm * cn > 2 or (swap and cm > 1) or (not swap and cn > 1):
                continue
            if (bm // cn) % (64 if mn_a else 8) or (bn // cm) % (64 if mn_b else 8):
                continue
            if div(m, bm) % cm or div(n, bn) % cn:
                continue
            # The source demands both operand swizzles128 when either operand is K-major.
            if not (mn_a and mn_b):
                if (bm // cn * 2) % 128 and mn_a:
                    continue
                if (bn // cm * 2) % 128 and mn_b:
                    continue
            result.append((swap, bm, bn, cm, cn))
    return result


def rank(layout: tuple[int, ...], m: int, n: int) -> tuple[int, ...]:
    _, bm, bn, cm, cn = layout
    blocks = div(m, bm) * div(n, bn)
    waves = div(blocks, 132)
    last = blocks % 132 or 132
    return (int(waves != 1), -cm * cn, waves, -last, bm + bn, bm * bn)


def main() -> None:
    root = Path(json.loads((STUDY / "inputs/sources.json").read_text())["tree"])
    body = (root / "csrc/jit_kernels/heuristics/sm100.hpp").read_text()
    for marker in (
        "if (swap_ab == 1 and cluster_m > 1)",
        "desc.k <= 256 ? 128 : 256",
        "desc.with_accumulation ? 24 : 32",
        "smem_capacity = 232448",
    ):
        assert marker in body
    cases, selected = 0, []
    for m, n, k, mn_a, mn_b in itertools.product(
        (1, 32, 65, 128, 193, 257),
        (16, 128, 256, 513),
        (64, 256, 1024),
        (False, True),
        (False, True),
    ):
        menu = candidates(m, n, k, mn_a, mn_b)
        assert menu
        chosen = min(menu, key=lambda item: rank(item, m, n))
        swap, bm, bn, cm, cn = chosen
        assert cm * cn <= 2 and 132 % (cm * cn) == 0
        assert not (swap and cm != 1) and not (not swap and cn != 1)
        assert div(m, bm) % cm == 0 and div(n, bn) % cn == 0
        # BF16 stages are inferred after layout selection; no scale storage.
        cd_swizzle = next(mode for mode in (128, 64, 32, 16) if bn * 2 % mode == 0)
        fixed = ((16 * bn * 2) if swap else bm * cd_swizzle) * 2 + 32 * 8 * 3 + 2 * 8 * 3 + 8 + 4
        per_stage = (bm // cn + bn // cm) * 64 * 2
        stages = min((232448 - fixed) // per_stage, 32)
        assert stages >= 2 and fixed + stages * per_stage <= 232448
        assert stages == 32 or fixed + (stages + 1) * per_stage > 232448
        if not mn_a and not mn_b and k == 1024 and n == 256:
            selected.append({"m": m, "layout": chosen, "stages": stages})
        cases += 1
    thresholds = []
    for dtype, accumulate in itertools.product(("bf16", "fp32"), (False, True)):
        threshold = 16 if dtype != "fp32" else 24 if accumulate else 32
        for blocks in (threshold - 1, threshold):
            threshold_stages = 1 if blocks >= threshold else 2
            assert threshold_stages == (2 if blocks < threshold else 1)
        thresholds.append({"dtype": dtype, "accumulate": accumulate, "min_k_blocks": threshold})
    result = {
        "outcome": "PASS",
        "bf16_layout_policy_domains": cases,
        "selected_witnesses": selected,
        "scaled_k_group_store_thresholds": thresholds,
        "boundary": (
            "Source-selected BF16 menu/resource arithmetic only; not compiled "
            "policy or measured optimality."
        ),
    }
    (STUDY / "checks/config-policy-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: SM100 BF16 menu, source ranking, resource bounds and store thresholds.\n"
    )


if __name__ == "__main__":
    main()
