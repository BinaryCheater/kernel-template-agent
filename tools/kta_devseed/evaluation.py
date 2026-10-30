"""Evaluation records, pool snapshots and coverage reports for the WP9a fixture store.

# kta:fixture synthetic Bound/Candidate/Prediction/Build/Check/Measurement/Solution records and
coverage Reports (D95, D168). No number here was measured.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kta.core.schemas import Actor
from kta_devseed.knowledge import SYSTEM, Seeder

Go = Callable[[float], None]


def _tool(name: str) -> Actor:
    return Actor(type="tool", name=name)


def _checker(name: str) -> Actor:
    return Actor(type="checker", name=name)


def _job_refs(s: Seeder, extra: dict[str, str]) -> dict[str, str]:
    return {"env": s.ids["env:spark"], "worker": s.ids["worker:spark"], **extra}


def _measure(s: Seeder, key: str, us: float, refs: dict[str, str]) -> str:
    m = s.put(
        key,
        "Measurement",
        {
            "median_us": us,
            "ci_low_us": round(us * 0.993, 2),
            "ci_high_us": round(us * 1.009, 2),
            "n": 200,
            "protocol": "measure-v1",
            "clocks": {"sm_mhz": 2418},
            "temps": {"gpu_c": 51},
        },
        refs=_job_refs(s, refs),
        actor=_tool("kta measure"),
    )
    s.status(m, _checker("measure_protocol@1"), to_status="reproduced")
    return m


def _candidates(
    s: Seeder,
    case: str,
    family: str,
    rows: list[tuple[dict[str, Any], float, float | None]],
    per_unit: dict[str, float],
) -> list[tuple[str, str | None, str | None, str | None]]:
    out = []
    for i, (assignment, pred, meas) in enumerate(rows):
        c = s.put(
            f"cand:{case}:{i}",
            "Candidate",
            {"assignment": assignment, "bucket": "all"},
            refs={"case": s.ids[f"case:{case}"], "family": s.ids[family]},
            actor=_tool("kta solve"),
        )
        s.put(
            f"pred:{case}:{i}",
            "Prediction",
            {"per_unit": per_unit, "serial": 1.0, "total": pred, "model": "gb10@2"},
            refs={"candidate": c},
            actor=_tool("kta solve"),
        )
        if meas is None:
            out.append((c, None, None, None))
            continue
        b = s.put(
            f"build:{case}:{i}",
            "Build",
            {
                "ok": True,
                "regs": 18 if case == "vadd" else 96,
                "spills": 0,
                "smem": None if case == "vadd" else 16384,
                "log_summary": "ptxas: 0 bytes spill",
            },
            refs=_job_refs(s, {"candidate": c}),
            actor=_tool("nvcc 13.0.88"),
        )
        sass = s.put(
            f"sass:{case}:{i}",
            "Check",
            {"passed": True, "failures": []},
            refs=_job_refs(s, {"build": b}),
            actor=_checker("check_sass@1"),
        )
        num = s.put(
            f"num:{case}:{i}",
            "Check",
            {"passed": True, "failures": []},
            refs=_job_refs(s, {"build": b}),
            actor=_checker("numeric_checker@3"),
        )
        for x in (sass, num):
            s.status(x, _checker("checker"), to_status="reproduced")
        m = _measure(s, f"meas:{case}:{i}", meas, {"build": b})
        out.append((c, b, num, m))
    return out


def seed_evaluation(s: Seeder, go: Go) -> None:
    # vadd: accepted in round 1 by rule 3 (beats the strongest reference, gap explained).
    go(11.3)
    s.put(
        "bound:vadd",
        "Bound",
        {"per_unit_us": {"dram": 41.9, "launch": 0.0}, "bound_us": 41.9, "bottleneck": "dram"},
        refs={"case": s.ids["case:vadd"]},
        actor=_tool("kta bound"),
    )
    go(17)
    for name, us in (("torch eager", 47.8), ("cub::DeviceTransform", 44.9)):
        _measure(
            s,
            f"base:vadd:{name}",
            us,
            {"case": s.ids["case:vadd"], "baseline": s.ids[f"baseline:{name}"]},
        )
    rows = _candidates(
        s,
        "vadd",
        "family:vadd_f32@1",
        [
            ({"block": 256, "unroll": 2, "vec": 4}, 42.1, 43.6),
            ({"block": 512, "unroll": 1, "vec": 4}, 42.3, 43.9),
            ({"block": 256, "unroll": 1, "vec": 4}, 42.4, 44.0),
            ({"block": 128, "unroll": 2, "vec": 4}, 42.6, 44.4),
        ],
        {"dram": 41.0, "launch": 1.1},
    )
    go(22.5)
    best = rows[0]
    sol = s.put(
        "solution:vadd",
        "Solution",
        {
            "bucket": "n=2^24",
            "assignment": {"block": 256, "unroll": 2, "vec": 4},
            "pct_of_bound": 41.9 / 43.6,
        },
        refs={
            "case": s.ids["case:vadd"],
            "family": s.ids["family:vadd_f32@1"],
            "measurement": best[3] or "",
            "check": best[2] or "",
            "bound": s.ids["bound:vadd"],
        },
        actor=SYSTEM,
    )
    s.put(
        "judge:vadd:1",
        "Report",
        {
            "title": "judgement",
            "subject": s.ids["case:vadd"],
            "summary": "Accepted by rule 3: faster than the strongest reference (44.9 µs); "
            "the remaining 1.7 µs is launch overhead and the tail iteration.",
            "data": {
                "kind": "judgement",
                "round": 1,
                "rule": 3,
                "explained_share": 0.88,
                "units": [
                    {"unit": "dram", "bound_us": 41.9, "predicted_us": 41.0, "measured_us": 42.3},
                    {"unit": "launch", "bound_us": 0.0, "predicted_us": 1.1, "measured_us": 1.3},
                ],
            },
        },
        refs={"case": s.ids["case:vadd"], "solution": sol},
    )

    # gemm: round 1 judged by rule 4 (gap unexplained), a Gap and a T1 follow.
    go(16)
    s.put(
        "bound:gemm",
        "Bound",
        {
            "per_unit_us": {"launch": 3.1, "smem": 0.4, "tensor": 0.3, "dram": 0.2},
            "bound_us": 3.1,
            "bottleneck": "launch",
        },
        refs={"case": s.ids["case:gemm"]},
        actor=_tool("kta bound"),
    )
    go(28)
    for name, us in (("torch.matmul (cuBLAS)", 6.4),):
        _measure(
            s,
            f"base:gemm:{name}",
            us,
            {"case": s.ids["case:gemm"], "baseline": s.ids[f"baseline:{name}"]},
        )
    _candidates(
        s,
        "gemm",
        "family:gemm_tiled@1",
        [
            ({"BM": 64, "BN": 64, "BK": 32}, 6.1, 9.7),
            ({"BM": 64, "BN": 32, "BK": 32}, 6.3, 10.2),
            ({"BM": 32, "BN": 64, "BK": 32}, 6.4, 10.4),
            ({"BM": 128, "BN": 64, "BK": 32}, 6.6, 12.8),
        ],
        {"tensor": 0.9, "smem": 1.6, "dram": 0.6, "launch": 3.0},
    )
    go(34)
    s.put(
        "judge:gemm:1",
        "Report",
        {
            "title": "judgement",
            "subject": s.ids["case:gemm"],
            "summary": "Rule 4: 38% of the gap to the bound is not explained by the model; "
            "extend the family (T1).",
            "data": {"kind": "judgement", "round": 1, "rule": 4, "explained_share": 0.62},
        },
        refs={"case": s.ids["case:gemm"]},
    )
    s.put(
        "gap:gemm",
        "Gap",
        {
            "missing": "template",
            "description": "Global-load latency exposed in the K loop; references overlap "
            "loads with mma via cp.async pipelining",
            "level": 3,
            "attribution": "space",
        },
        refs={"case": s.ids["case:gemm"]},
    )

    # softmax: round 2 candidates predicted and built, measurement running.
    go(4)
    s.put(
        "bound:softmax",
        "Bound",
        {"per_unit_us": {"dram": 7.9}, "bound_us": 7.9, "bottleneck": "dram"},
        refs={"case": s.ids["case:softmax"]},
        actor=_tool("kta bound"),
    )
    for name, us in (("torch eager", 14.2), ("triton tutorial", 9.8)):
        _measure(
            s,
            f"base:softmax:{name}",
            us,
            {"case": s.ids["case:softmax"], "baseline": s.ids[f"baseline:{name}"]},
        )
    go(34)
    _candidates(
        s,
        "softmax",
        "family:softmax_rowwise@2",
        [
            ({"threads": 256}, 8.6, None),
            ({"threads": 512}, 8.7, None),
            ({"threads": 128}, 8.9, None),
        ],
        {"dram": 8.1},
    )
    go(44)
    s.put(
        "bound:rmsnorm",
        "Bound",
        {"per_unit_us": {"dram": 30.7}, "bound_us": 30.7, "bottleneck": "dram"},
        refs={"case": s.ids["case:rmsnorm"]},
        actor=_tool("kta bound"),
    )


def seed_snapshots(s: Seeder, ops: Any, go: Go) -> None:
    """Pool snapshots written like ``kta.store.snapshot`` but with origin "fixture" (that helper
    always writes origin "real"), then moved onto ``current_pool`` by compare-and-set."""
    fam = [s.ids[k] for k in s.ids if k.startswith("family:")]
    tpl = [s.ids[k] for k in s.ids if k.startswith("template:")]
    early = [s.ids["family:vadd_f32@1"], s.ids["family:gemm_tiled@1"]]
    plan = [
        (-55.0, "pool@1 seed", early + tpl[:9], 5),
        (-20.0, "pool@2", early + tpl[:11], 7),
        (32.0, "pool@3", fam + tpl, 9),
    ]
    parent: str | None = None
    for i, (minute, note, members, n) in enumerate(plan):
        go(minute)
        refs: dict[str, Any] = {"members": sorted(set(members))}
        if parent:
            refs["previous"] = parent
        sid = s.put(
            f"snapshot:{i}",
            "PoolSnapshot",
            {"members": sorted(set(members)), "note": note},
            refs=refs,
        )
        if not ops.set_pointer("current_pool", parent, sid):
            raise ValueError("current_pool moved while seeding")
        s.put(
            f"coverage:{i}",
            "Report",
            {
                "title": "coverage",
                "subject": sid,
                "summary": f"coverage of {note}",
                "data": _coverage(i, n),
            },
        )
        parent = sid


MEAN = [5, 7.6, 9.4, 10.6, 11.5, 12.1, 12.6, 13.0, 13.2]
SPREAD = [0, 1.1, 1.4, 1.5, 1.4, 1.2, 1.0, 0.7, 0]


def _coverage(i: int, n: int) -> dict[str, Any]:
    grid = [
        [2, 1, 0, 3, 2, 0],
        [0, 0, 0, 1 if i == 2 else 0, 0, 0],
        [0] * 6,
        [0, 0, 1, 0, 0, 0],
        [0, 2, 0, 0, 0, 1],
        [0] * 6,
    ]
    best = [
        ["tested", "tested", "", "tested", "tested", ""],
        ["", "", "", "declared", "", ""],
        [""] * 6,
        ["", "", "tested", "", "", ""],
        ["", "bounded", "", "", "", "tested"],
        [""] * 6,
    ]
    return {
        "scope": {
            "description": "public kernel tutorials and libraries (toy fixture)",
            "total": 9,
            "by_category": {"tutorials": 4, "libraries": 3, "LLM inference kernels": 2},
        },
        "discovery": {
            "sources": list(range(1, n + 1)),
            "mean": MEAN[:n],
            "lo": [m - d for m, d in zip(MEAN[:n], SPREAD, strict=False)],
            "hi": [m + d for m, d in zip(MEAN[:n], SPREAD, strict=False)],
            "chao1": [
                {"estimate": 11, "lo": 9, "hi": 19},
                {"estimate": 15, "lo": 12, "hi": 24},
                {"estimate": 16, "lo": 13.5, "hi": 23},
            ][i],
        },
        "trend": [
            {
                "snapshot": f"pool@{k + 1}",
                "good_turing": [0.44, 0.61, 0.72][k],
                "expressible": [0.25, 0.5, 0.67][k],
            }
            for k in range(i + 1)
        ],
        "instruction_families": [
            {
                "family": "LDG/STG.128 vectorized global access",
                "we_produce": True,
                "sota_uses": True,
                "note": "vector_load@2, coalesced_store@1",
            },
            {
                "family": "SHFL warp shuffle",
                "we_produce": True,
                "sota_uses": True,
                "note": "warp_reduce@1",
            },
            {
                "family": "HMMA bf16 (mma.sync m16n8k16)",
                "we_produce": True,
                "sota_uses": True,
                "note": "mma_bf16@1",
            },
            {
                "family": "LDGSTS (cp.async)",
                "we_produce": i == 2,
                "sota_uses": True,
                "note": "smem_double_buffer@1 (declared)" if i == 2 else "no Template",
            },
            {
                "family": "LDSM (ldmatrix)",
                "we_produce": False,
                "sota_uses": True,
                "note": "T3 escalated",
            },
            {
                "family": "UTMALDG (TMA bulk copy)",
                "we_produce": False,
                "sota_uses": True,
                "note": "support on sm_121 to probe",
            },
        ],
        "hw_patterns": {
            "features": ["ldg_128", "cp_async", "tma", "mma_sync_bf16", "shfl_sync", "atomics"],
            "patterns": ["map", "reduce", "contract", "transfer", "store", "sync"],
            "cells": [
                {"f": f, "p": p, "n": grid[f][p], "best": best[f][p]}
                for f in range(6)
                for p in range(6)
            ],
        },
        "references": [
            {
                "name": "cub_block_reduce",
                "state": ["frozen", "reproduced", "covered"][i],
                "gaps": 0,
            },
            {
                "name": "cutlass_sgemm_example",
                "state": "frozen" if i == 0 else "gap",
                "gaps": 0 if i == 0 else 3,
            },
            {
                "name": "triton_softmax_tutorial",
                "state": "analyzing" if i == 2 else "not started",
                "gaps": 0,
            },
        ],
        "gap_explained": [
            {"case": "vadd_f32", "explained": 0.88, "unexplained": 0.12},
            {"case": "gemm_bf16_256", "explained": 0.62, "unexplained": 0.38},
        ],
        "backlog": [
            {
                "item": "flashinfer_decode",
                "why": "Uses cp.async and ldmatrix, both red rows",
                "priority": 1,
            },
            {
                "item": "T3 ldmatrix swizzle (escalated)",
                "why": "Blocks cutlass_sgemm_example and the gemm gap",
                "priority": 2,
            },
        ],
        "structure": {
            "other_patterns": 1,
            "unmatched_by_kind": {"layout": 2, "epilogue": 1},
            "assurance": {"declared": 1, "tested": 10, "bounded": 1},
        },
    }
