"""Round-1 probes as regressions (docs/design/perfmodel/LOOP.md product 5): the replay of a probe's
own SASS stays within tolerance of what batch r1 measured on the GB10."""

from __future__ import annotations

from pathlib import Path

import pytest

from kta.models.gb10sim import probeset
from kta.models.gb10sim.probes import Case

R1 = Path(__file__).resolve().parents[2] / "perfmodel" / "batches" / "r1"
POINTS = {  # id: tolerance
    "shfl latency_w1": 0.05,
    "shfl tput_w32": 0.05,
    "smem_ldsts LDS.128_w32": 0.05,
    "smem_ldsts STS.32_w32": 0.05,
    "namedbar arrive_L256": 0.05,
    "namedbar sync_w8": 0.05,
    "store_tail tl_b48": 0.05,
}


@pytest.fixture(scope="module")
def cases() -> dict[str, Case]:
    return {c.id: c for c in probeset.all_cases(R1)}


@pytest.mark.parametrize("cid", sorted(POINTS))
def test_round1_probe_point_replays_within_tolerance(cases, cid: str) -> None:
    case = cases[cid]
    predicted, _, _ = probeset.run_case(case)
    assert abs(predicted / case.measured - 1) <= POINTS[cid], (predicted, case.measured)


def test_bar_arrive_does_not_wait_for_the_consumers(cases) -> None:
    short = probeset.run_case(cases["namedbar arrive_L0"])[0]
    long = probeset.run_case(cases["namedbar arrive_L256"])[0]
    assert long == pytest.approx(short, abs=1.0)


R3 = R1.parent / "r3"
R3_POINTS = {  # round 3 (batch r3): warm-L2 residency and launch, id: tolerance
    "l2reuse n16_ca_cold": 0.05,
    "l2reuse n16_ca_warm": 0.1,
    "l2reuse n16_cs_warm": 0.05,
    "l2reuse n43_ca_warm": 0.1,
    "launch work_i16384": 0.25,
}


@pytest.fixture(scope="module")
def cases_r3() -> dict[str, Case]:
    return {c.id: c for c in probeset.all_cases(R3)}


@pytest.mark.parametrize("cid", sorted(R3_POINTS))
def test_round3_probe_point_replays_within_tolerance(cases_r3, cid: str) -> None:
    case = cases_r3[cid]
    predicted, _, _ = probeset.run_case(case)
    assert abs(predicted / case.measured - 1) <= R3_POINTS[cid], (predicted, case.measured)


def _counters(unit: str) -> dict:
    import json

    return json.loads((R3 / "measured.json").read_text())["counters"][unit]


@pytest.mark.parametrize(
    "unit", ["gemv_w4a16_n4096_k4096_warm", "rmsnorm_m512_n4096_warm", "softmax_m512_n4096_warm"]
)
def test_ldg_sectors_from_lane_addresses_match_the_profiler(cases_r3, unit: str) -> None:
    """R1 with sectors (D765): global-load sectors per block from the lanes' addresses, and
    instructions per block (FCHK fast path, D770), within 2 % of ncu."""
    _, sm, _ = probeset.run_case(cases_r3[f"{unit} kernel"])
    c = _counters(unit)
    grid = c["launch__grid_size"]
    sect = c["l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"] / grid
    inst = c["smsp__inst_executed.sum"] / grid
    assert sm.count["ldg_sectors"] / sm.blocks == pytest.approx(sect, rel=0.02)
    assert sm.count["issued"] / sm.blocks == pytest.approx(inst, rel=0.02)
