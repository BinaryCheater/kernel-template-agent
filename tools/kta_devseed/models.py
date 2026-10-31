"""Fixture data for the hardware-model scoreboard and the predicted-vs-measured timeline
(WP16, D510).

The predictions are real L2 runs on the seed GB10 model (knowledge/seed/hardware) over
hand-written GEMM effects (``gemm_effects``, the transactions of the seed GEMM chain in
closed form). The "measured" side is L2 on a deliberately different model (L2 peak x 0.85)
plus a fixed +-3 % pattern. Every record has origin ``fixture``.
"""

# kta:fixture synthetic "measured" times for the scoreboard; no kernel has been measured yet

from __future__ import annotations

import math
from typing import Any

import yaml

from kta.core.config import repo_root
from kta.core.controls import baseline
from kta.core.schemas.hardware import HwModel
from kta.core.schemas.hwmodel import HwArch, ModelVersion
from kta.models.l1 import EvalEffects, EvalOverlap, EvalStorage, EvalTx
from kta.models.l2 import moved, simulate
from kta.models.machine import MachineModel
from kta.models.regression import Observation, admit, results, summarise
from kta_devseed.knowledge import Seeder

HARDWARE = repo_root() / "knowledge" / "seed" / "hardware"
KERNELS: dict[str, tuple[dict[str, int], list[dict[str, int]], bool]] = {
    "gemm_fp16 4096^3": (
        {"M": 4096, "N": 4096, "K": 4096},
        [
            {"BM": 128, "BN": 128, "BK": 32, "S": 3},
            {"BM": 128, "BN": 64, "BK": 32, "S": 2},
            {"BM": 64, "BN": 64, "BK": 32, "S": 4},
            {"BM": 128, "BN": 256, "BK": 32, "S": 2},
        ],
        False,
    ),
    "gemm_fp16 128x128x16384": (
        {"M": 128, "N": 128, "K": 16384},
        [
            {"BM": 64, "BN": 64, "BK": 64, "S": 2},
            {"BM": 128, "BN": 128, "BK": 32, "S": 3},
            {"BM": 32, "BN": 64, "BK": 128, "S": 2},
        ],
        True,
    ),
}
NOISE = (1.03, 0.98, 1.01, 0.97)


def gb10() -> MachineModel:
    def load(name: str) -> Any:
        return yaml.safe_load((HARDWARE / name).read_text(encoding="utf-8"))

    return MachineModel.build(
        HwModel.model_validate(load("model_gb10.yaml")),
        HwArch.model_validate(load("arch_sm12x.yaml")),
    )


def gemm_effects(shape: dict[str, int], c: dict[str, int]) -> EvalEffects:
    """The seed GEMM chain's transactions in closed form (FP16, cp.async, mma.sync)."""
    m, n, k = shape["M"], shape["N"], shape["K"]
    bm, bn, bk, s = c["BM"], c["BN"], c["BK"], c["S"]
    ctas = math.ceil(m / bm) * math.ceil(n / bn)
    trips = math.ceil(k / bk)
    tile_bytes = (bm + bn) * bk * 2
    txs = (
        EvalTx("tiles", "ctas", "launch", "cta_launch", ctas),
        EvalTx("tiles", "kernel", "launch", "kernel_launch", 1),
        EvalTx(
            "tiles",
            "compulsory",
            "transfer",
            "dram_stream",
            1,
            {"bytes": 2.0 * (m * k + k * n + m * n)},
        ),
        EvalTx(
            "ab_async",
            "copy",
            "transfer",
            "cp_async_16",
            ctas * trips * tile_bytes / 16,
            {"bytes": 16.0, "vector_bytes": 16.0},
        ),
        EvalTx("ab_stages", "barrier", "sync", "cta_barrier", ctas * trips, {"warps": 4.0}),
        EvalTx(
            "ab_swizzle",
            "access",
            "access",
            "smem_access",
            ctas * trips * tile_bytes * (bm // 16 + bn // 16) / 2048,
            {"wavefronts": 4.0},
        ),
        EvalTx(
            "mma",
            "mma",
            "contract",
            "mma_sync_m16n8k16_f16",
            ctas * bm * bn * trips * bk * 2 / 4096,
            {"flops": 4096.0},
        ),
    )
    storage = (EvalStorage("ab_smem", "buffer", "smem", "cta", float(s * tile_bytes)),)
    overlaps = (
        EvalOverlap("ab_stages", ("pattern:transfer", "pattern:contract"), True, s - 1, trips),
    )
    return EvalEffects(txs, storage, overlaps)


def _truth(mm: MachineModel) -> MachineModel:
    return moved(mm, peaks={"l2": (mm.resources["l2"].rate or 1.0) * 0.85})


def seed_models(seeder: Seeder) -> list[str]:
    """Two ModelVersion records with scored regression results; returns their IDs."""
    base = gb10()
    truth = _truth(base)
    versions = [
        ("gb10 L2 v1", base, "initial", "seed GB10 instance (documented values)"),
        (
            "gb10 L2 v2",
            moved(base, peaks={"l2": (base.resources["l2"].rate or 1.0) * 0.9}),
            "machine_model_change",
            "fixture: L2 bandwidth lowered after the largest-error case",
        ),
    ]
    out: list[str] = []
    previous: str | None = None
    for label, mm, change, reason in versions:
        obs: list[Observation] = []
        for kernel, (shape, configs, holdout) in KERNELS.items():
            for i, c in enumerate(configs):
                eff = gemm_effects(shape, c)
                pred = simulate(eff, mm)
                meas = simulate(eff, truth)
                f = NOISE[i % len(NOISE)]
                rid = seeder.put(
                    f"l2:{label}:{kernel}:{i}",
                    "Report",
                    {
                        "title": "l2_simulation",
                        "subject": pred.model,
                        "summary": f"fixture: {kernel} {c} predicted {pred.total_us:.1f} us",
                        "data": {**pred.summary(), "kernel": kernel, "configuration": c},
                    },
                )
                obs.append(
                    Observation(
                        kernel,
                        c,
                        pred.total_us,
                        meas.total_us * f,
                        pred.busy_us,
                        {r: b * f for r, b in meas.busy_us.items()},
                        holdout,
                        rid,
                        None,
                    )
                )
        res = results(obs)
        summary = summarise(res)
        body = ModelVersion(
            target="gb10",
            label=label,
            hw_model="gb10@1",
            arch="sm12x@1",
            simulators={"L1": "fixture", "L2": "fixture"},
            change=change,  # type: ignore[arg-type]
            reason=reason,
            results=res,
            summary=summary,
            admitted=admit(summary, baseline().version_admission),
        )
        refs = {"previous": previous} if previous else {}
        previous = seeder.put(
            f"model_version:{label}", "ModelVersion", body.model_dump(mode="json"), refs
        )
        out.append(previous)
    return out
