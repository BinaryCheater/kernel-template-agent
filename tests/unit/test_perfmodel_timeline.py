"""Scope's exports: a recorded replay is the replay, a trace adds up to the engine's own totals, and
the atlas reads the inventory's tables as written."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kta.models.gb10sim import probeset
from kta.models.gb10sim.sm import SM

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from kta_perfmodel import atlas, table, timeline

R1 = Path(__file__).resolve().parents[2] / "perfmodel" / "batches" / "r1"
ROW = "shfl tput_w32"


def job() -> list[str]:
    return ["batch", "r1", ROW]


def replayed() -> bool:
    return (R1 / "build").is_dir()


@pytest.mark.skipif(not replayed(), reason="batch r1 is local (D914)")
def test_recording_changes_nothing_and_leaves_the_engine_as_it_was() -> None:
    before = SM.issue, SM.ev
    case = {c.id: c for c in probeset.all_cases(R1)}[ROW]
    plain, *_ = probeset.run_cases([case], trace=True)
    rec: dict = {}
    with timeline.recording(rec):
        kept, *_ = probeset.run_cases([case], trace=True)
    assert kept == plain
    assert (SM.issue, SM.ev) == before
    assert any(r["holds"] for r in rec.values())


@pytest.mark.skipif(not replayed(), reason="batch r1 is local (D914)")
def test_a_trace_adds_up_to_the_engine() -> None:
    from kta.models.gb10sim.machine import default_machine

    spec = {"id": ROW, "job": job(), "group": "probe shfl", "unit": "clk"}
    d = timeline.build(spec, Path("."), {}, 0, None)
    c = d["meta"]["checks"]
    assert c["path_sum"] == pytest.approx(c["cycles"], abs=1.0)
    assert len(d["issues"]["t"]) == d["meta"]["issued"]
    assert d["issues"]["t"] == sorted(d["issues"]["t"])
    assert d["machine"]["params"]
    assert default_machine().status("sim.hmma_16816_pipe_clk") == "measured"
    for p in c["pipes"].values():
        assert p["trace"] <= p["engine"] + 1.0


def test_a_hold_belongs_to_its_instruction_or_is_a_delay_beside_it() -> None:
    cls = SimpleNamespace(hold=(("fma", "x"),), either=())
    other = SimpleNamespace(hold=(), either=())
    sm = SimpleNamespace(icls=[cls, other])
    assert timeline._own(sm, 0, "fma")
    assert not timeline._own(sm, 0, "tensor")
    assert not timeline._own(sm, 1, "tensor")
    assert timeline._own(sm, 1, "mio")


def test_short_stretches_between_barriers_join_the_one_before() -> None:
    ins = [SimpleNamespace(op="NOP") for _ in range(60)]
    names = ["alu"] * 60
    names[20] = names[25] = names[50] = "barrier"
    regs = timeline.regions(ins, names)
    assert [(r["start"], r["end"]) for r in regs] == [(0, 20), (20, 50), (50, 60)]


INVENTORY = """
## 1. Issue

| Mechanism | Sources | Status | Kernels | Var |
| --- | --- | --- | --- | --- |
| `mma` pipe | `mma_pair` | Round 6 (probe mma_pipe, D783): M: 16 **structural** | G | H |
| Yield bit | gb10 | X (parsed, ignored) | all | — |

## 7a. Coverage matrix (LOOP §3a)

**read (LDG, LDS) × shared**: decides layout.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | M | m | · |  | smem_ldsts (D784) |

## 8. Counts

| Mechanism | Sources | Status | Kernels | Var |
| --- | --- | --- | --- | --- |
| Not a row | - | - | - | - |
"""


def test_inventory_rows_keep_their_state_probes_and_decisions() -> None:
    rows = atlas.inventory(INVENTORY)
    assert [r["name"] for r in rows] == ["mma pipe", "Yield bit"]
    first, second = rows
    assert (first["code"], first["depth"], first["decisions"]) == ("M", "structural", ["D783"])
    assert first["rounds"] == [6]
    assert "mma_pipe" in first["probes"] and "mma_pair" in first["probes"]
    assert "tensor" in first["resources"]
    assert second["code"] == "X"


def test_the_matrix_keeps_operation_target_and_cells() -> None:
    (g,) = atlas.matrix(INVENTORY)
    assert (g["op"], g["target"]) == ("read", "shared")
    assert g["rows"][0]["w"] == "M" and g["rows"][0]["b"] == "m"


def test_rule_code_comes_from_the_engine_itself() -> None:
    src = Path(table.ENGINE / "sm.py").read_text()
    rules = atlas.rule_code(src, {"pipe", "cp_async"})
    assert rules["pipe"]["code"].startswith("    def _op_pipe")
    # through _return_stage-style helpers too: bound from the code, not by hand
    assert "sim.mio_ldgsts_clk" in rules["cp_async"]["reads"]
