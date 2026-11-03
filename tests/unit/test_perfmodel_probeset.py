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
