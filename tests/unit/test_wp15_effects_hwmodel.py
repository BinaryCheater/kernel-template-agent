"""WP15: Template effects, the declarative hardware model and L1 (spec/workplan.md WP15).

Covers the code-stage acceptance: seed Templates validate with effects; the GEMM seed
chain is priced by L1 with per-unit SOL on the GB10 seed model; changing one wall or one
peak changes the prediction without code; schema-1 cost records upcast; unpriced
operation classes and unmodelled resources are counted, never silently zero-cost.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from knowhelp import T0, seed, source_id
from kta.core.config import repo_root
from kta.core.context import Context
from kta.core.errors import NotSupported, SchemaError
from kta.core.ids import record_id
from kta.core.schemas import Actor, Record
from kta.core.schemas.evaluation import Case, Problem
from kta.core.schemas.hardware import HwModel
from kta.core.schemas.hwmodel import HwArch, ModelVersion, Provenance, RegressionSet
from kta.core.schemas.knowledge import Template
from kta.formulation import assemble, per_unit_work, price_point, work_terms
from kta.knowledge import load_family, validate_template
from kta.knowledge.pool import find_template, templates
from kta.measure import interference
from kta.models import (
    EvalEffects,
    EvalOverlap,
    EvalStorage,
    EvalTx,
    MachineModel,
    NoResidentCta,
    UnitModel,
    bound,
    latest_machine,
    min_work,
    price,
)
from kta.store.base import Store
from kta_wp7.pipeline import templates_of

SEED = repo_root() / "knowledge" / "seed"
HARDWARE = SEED / "hardware"
FAMILY = SEED / "families" / "gemm_fp16_tilelang"
SHAPE = {"M": 4096, "N": 4096, "K": 4096}
POINT = {"BM": 128, "BN": 128, "BK": 32, "S": 3, "P": 2, "threads": 128, "split": 1, "KT": 128}


def _yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def gb10() -> tuple[HwArch, HwModel]:
    arch = HwArch.model_validate(_yaml(HARDWARE / "arch_sm12x.yaml"))
    return arch, HwModel.model_validate(_yaml(HARDWARE / "model_gb10.yaml"))


def gemm_case() -> tuple[Problem, Case]:
    doc = _yaml(SEED / "cases" / "gemm_fp16.yaml")
    case = Case(problem="gemm_fp16", target="gb10", **doc["case"])
    return Problem.model_validate(doc["problem"]), case


def gemm_formulation(ctx: Context, store: Store, mm: MachineModel) -> Any:
    seed(ctx, store)
    fp = load_family(FAMILY)
    return assemble(fp, fp.chain, templates_of(store, fp), gemm_case()[1], mm.hw)


# ---- schema, seed and upcast ---------------------------------------------------------


def test_every_seed_template_validates_with_effects(ctx: Context, store: Store) -> None:
    files = [*SEED.glob("templates/*.yaml"), *SEED.glob("references/*/templates/*.yaml")]
    assert len(files) == 37
    for path in files:
        body = _yaml(path)
        assert "cost" not in body and "effects" in body, path
        t = Template.model_validate(body)
        assert t.effects.upcast_from is None and t.cost == []
    seed(ctx, store)
    for _, t in templates(store):
        errors = [i for i in validate_template(t, store) if i.severity == "error"]
        assert errors == [], (t.name, errors)


def test_schema_1_cost_record_upcasts_on_read_and_keeps_its_id(store: Store) -> None:
    body = _yaml(SEED / "templates" / "multi_buffer_rotation.yaml")
    del body["effects"]
    body["cost"] = [
        {"unit": "issue", "work": "beta * p.iterations", "overlap": "serial"},
        {"unit": "dram", "work": "p.size_bytes * p.iterations if P == 0 else 0"},
    ]
    stored = {
        "kind": "Template",
        "schema": 1,
        "created_at": "2026-10-30T00:00:00Z",
        "actor": {"type": "human", "name": "old", "task": None, "commit": None, "dirty": None},
        "refs": {},
        "body": body,
        "blobs": {},
        "origin": "real",
    }
    rec = Record.model_validate(stored)
    assert rec.body == body  # the stored body is never rewritten
    rid = store.put(rec)
    again = store.get(rid)
    assert record_id(again) == rid and again.schema_ == 1
    t = again.body_model()
    assert isinstance(t, Template) and t.effects.upcast_from == "cost"
    assert [(c.unit, c.overlap) for c in t.cost] == [("issue", "serial"), ("dram", "overlappable")]
    assert [x.kind for x in t.effects.transactions] == ["unit:issue", "unit:dram"]
    hit = find_template(store, "multi_buffer_rotation@1")
    assert hit is not None and hit[1].effects.upcast_from == "cost"


def test_cost_and_effects_together_are_refused() -> None:
    body = _yaml(SEED / "templates" / "async_transfer.yaml")
    body["cost"] = [{"unit": "issue", "work": "1"}]
    with pytest.raises(ValueError, match="both schema-1 cost and schema-2 effects"):
        Template.model_validate(body)


def test_overlap_selector_must_name_a_transaction() -> None:
    body = _yaml(SEED / "templates" / "async_transfer.yaml")
    body["effects"]["overlap"] = [{"between": ["nope"], "when": "False"}]
    with pytest.raises(ValueError, match="names no transaction"):
        Template.model_validate(body)


def test_seed_imports_the_gb10_class_and_instance(ctx: Context, store: Store) -> None:
    seed(ctx, store)
    rid, mm = latest_machine(store, "gb10")
    assert mm.arch == "sm12x@1" and store.get(rid).refs["arch"]
    assert {r for r, x in mm.resources.items() if x.rate is None} == {"launch"}
    hw = gb10()[1]
    statuses = {p.status for p in hw.provenance.values()}
    assert statuses <= {"documented", "default"}  # nothing is measured by KTA yet
    assert all(p.source for p in hw.provenance.values())


def test_instance_on_a_class_needs_provenance_for_every_value() -> None:
    _, hw = gb10()
    body = hw.model_dump(mode="json")
    del body["provenance"]["limits.sm_count"]
    with pytest.raises(ValueError, match="no provenance"):
        HwModel.model_validate(body)


# ---- L1 on the GEMM seed chain ----------------------------------------------------------


def test_gemm_seed_chain_priced_by_l1_with_per_unit_sol(ctx: Context, store: Store) -> None:
    mm = MachineModel.build(*reversed(gb10()))  # type: ignore[arg-type]  # (hw, arch)
    f = gemm_formulation(ctx, store, mm)
    res = price_point(f, POINT, SHAPE, mm)
    busy = res.composed.busy_us
    assert {"tensor", "l2", "dram", "issue", "smem"} <= set(busy)
    # The contraction and the operand traffic are declared once and priced (WP5 issue).
    flops = 2 * 4096**3
    assert busy["tensor"] >= flops / mm.rate("tensor") * 1e6  # type: ignore[operator]
    assert res.unpriced == {} and set(res.unmodelled) == {"launch"}  # reported (D465)
    assert res.feasible and res.residency is not None and res.residency >= 1
    problem, case = gemm_case()
    sol = bound(case, SHAPE, mm, min_work(problem, SHAPE))
    assert set(sol.per_unit_us) == {"tensor", "dram"}
    for unit, t in sol.per_unit_us.items():
        assert t <= busy[unit] * (1 + 1e-12)  # per-unit SOL is a lower bound of busy
    p = UnitModel(target="gb10").predict_l1(res, mm)
    assert p.total >= sol.bound_us and p.unpriced == {"resource:launch": res.unmodelled["launch"]}


def test_changing_one_peak_or_wall_changes_the_prediction(ctx: Context, store: Store) -> None:
    arch, hw = gb10()
    mm = MachineModel.build(hw, arch)
    f = gemm_formulation(ctx, store, mm)
    skinny = {"M": 128, "N": 128, "K": 16384}
    point = {**POINT, "KT": 512}
    base = price_point(f, point, skinny, mm).total_us
    slow = hw.model_copy(update={"limits": {**hw.limits, "dram_bytes_per_s": 1.0e9}})
    assert price_point(f, point, skinny, MachineModel.build(slow, arch)).total_us > base
    # A wall moved below the point's shared-memory use: no launch (a cliff, not a cost).
    walls = [
        w.model_copy(update={"limit": "8192"}) if w.name == "smem_per_cta" else w
        for w in arch.walls
    ]
    tight = MachineModel.build(hw, arch.model_copy(update={"walls": walls}))
    res = price_point(f, point, skinny, tight)
    assert not res.feasible
    with pytest.raises(NoResidentCta, match="smem_per_cta"):
        UnitModel(target="gb10").predict_l1(res, tight)


def test_mip_rows_match_l1_amounts(ctx: Context, store: Store) -> None:
    mm = MachineModel.build(*reversed(gb10()))  # type: ignore[arg-type]
    f = gemm_formulation(ctx, store, mm)
    over, serial = per_unit_work(f, POINT, SHAPE, mm)
    rows: dict[str, float] = {}
    for t in work_terms(f, mm):
        rows[t.unit] = rows.get(t.unit, 0.0) + float(t.work.eval(POINT, hw=mm.hw, case=SHAPE))
    rows = {u: v for u, v in rows.items() if v}  # a transaction whose `when` fails counts 0
    totals = {u: over.get(u, 0.0) + serial.get(u, 0.0) for u in {*over, *serial}}
    assert rows.keys() == totals.keys()
    for u, v in totals.items():
        assert rows[u] == pytest.approx(v, rel=1e-12)


# ---- composition, failure fixtures -------------------------------------------------------


def _toy() -> tuple[MachineModel, HwArch]:
    arch = HwArch.model_validate(
        {
            "name": "toy",
            "version": 1,
            "description": "test architecture",
            "scopes": {"chip": "1"},
            "resources": [
                {
                    "name": "mem",
                    "kind": "throughput",
                    "scope": "chip",
                    "amount": "bytes",
                    "peak": "hw.bw",
                    "source": "test",
                },
                {
                    "name": "alu",
                    "kind": "throughput",
                    "scope": "chip",
                    "amount": "ops",
                    "peak": "hw.ops",
                    "source": "test",
                },
                {
                    "name": "port",
                    "kind": "throughput",
                    "scope": "chip",
                    "amount": "uses",
                    "source": "test: no peak known",
                },
            ],
            "operation_classes": [
                {
                    "name": "load",
                    "reservation": [{"resource": "mem", "amount": "tx.bytes"}],
                    "latency_cycles": "100",
                    "source": "test",
                },
                {
                    "name": "op",
                    "reservation": [
                        {"resource": "alu", "amount": "1"},
                        {"resource": "port", "amount": "1"},
                    ],
                    "latency_cycles": "10",
                    "source": "test",
                },
                {
                    "name": "lut",
                    "reservation": [{"resource": "mem", "amount": "tx.bytes * tbl.penalty"}],
                    "source": "test",
                },
            ],
            "tables": [
                {
                    "name": "penalty",
                    "keys": ["tx.stride"],
                    "lookup": "floor",
                    "rows": [[1, 1.0], [4, 2.0]],
                    "source": "test",
                }
            ],
            "composition": {"clock_hz": "1e9"},
        }
    )
    hw = HwModel(
        target="toy",
        version=1,
        arch="toy@1",
        limits={"bw": 1e9, "ops": 1e9},
        provenance={
            k: Provenance(status="default", source="test") for k in ("limits.bw", "limits.ops")
        },
    )
    return MachineModel.build(hw, arch), arch


def _tx(i: str, pattern: str, kind: str, count: float, **amounts: float) -> EvalTx:
    return EvalTx("s", i, pattern, kind, count, amounts)


def test_pair_and_recurrence_bounds_and_serial_terms() -> None:
    mm, _ = _toy()
    txs = (_tx("ld", "transfer", "load", 1, bytes=1000.0), _tx("mac", "contract", "op", 600))
    sel = ("pattern:transfer", "pattern:contract")
    overlapped = price(EvalEffects(txs, (), (EvalOverlap("s", sel, True, 2.0, 10.0),)), mm)
    assert overlapped.total_us == pytest.approx(1.0)  # max(1 us mem, 0.6 us alu)
    assert overlapped.composed.bounds_us == {
        "recurrence s:pattern:transfer|pattern:contract": pytest.approx(10 * 110e-9 / 2 * 1e6)
    }
    serial = price(EvalEffects(txs, (), (EvalOverlap("s", sel, False),)), mm)
    assert serial.total_us == pytest.approx(1.6)  # the pair cannot overlap
    alone = price(EvalEffects(txs, (), (EvalOverlap("s", ("s.mac",), False),)), mm)
    assert alone.total_us == pytest.approx(1.6) and alone.prediction().serial == pytest.approx(0.6)


def test_unpriced_class_and_unmodelled_resource_are_counted_not_free() -> None:
    mm, _ = _toy()
    res = price(
        EvalEffects((_tx("x", "other", "made_up_op", 7), _tx("m", "contract", "op", 5))), mm
    )
    assert res.unpriced == {"made_up_op": 7} and res.unmodelled == {"port": 5}
    assert not res.complete
    assert res.prediction().unpriced == {"made_up_op": 7, "resource:port": 5}


def test_table_term_lookup_and_missing_row() -> None:
    mm, _ = _toy()
    res = price(EvalEffects((_tx("t", "access", "lut", 1, bytes=1000.0, stride=8.0),)), mm)
    assert res.composed.busy_us["mem"] == pytest.approx(2.0)  # floor row 4 -> penalty 2
    with pytest.raises(NotSupported, match="no row"):
        price(EvalEffects((_tx("t", "access", "lut", 1, bytes=1.0, stride=0.5),)), mm)


def test_unknown_reservation_resource_and_bad_expression_are_refused() -> None:
    _, arch = _toy()
    body = arch.model_dump(mode="json")
    body["operation_classes"][0]["reservation"][0]["resource"] = "nowhere"
    with pytest.raises(ValueError, match="not declared"):
        HwArch.model_validate(body)
    body = arch.model_dump(mode="json")
    body["resources"][0]["peak"] = "hw.bw +"
    with pytest.raises(ValueError, match=r"resources\.mem\.peak"):
        HwArch.model_validate(body)


def test_walls_lower_occupancy_gives_residency() -> None:
    arch, hw = gb10()
    mm = MachineModel.build(hw, arch)
    storage = (EvalStorage("s", "buf", "smem", "cta", 49152.0),)
    res = price(EvalEffects((), storage), mm)
    by_name = {w.name: w for w in res.walls}
    assert by_name["smem_per_cta"].ok
    assert by_name["smem_per_sm"].residency == 102400 // (49152 + 1024)


def test_flat_schema_1_hwmodel_prices_only_legacy_cost() -> None:
    flat = HwModel(target="t", version=1, peaks={"dram": 1e9})
    mm = MachineModel.build(flat)
    res = price(EvalEffects((_tx("c", "other", "unit:dram", 1e3), _tx("x", "x", "load", 1))), mm)
    assert res.composed.busy_us == {"dram": pytest.approx(1.0)} and res.unpriced == {"load": 1}
    with pytest.raises(NotSupported, match="needs its HwArch"):
        MachineModel.build(gb10()[1])


def test_unpriced_kind_is_a_validation_warning(ctx: Context, store: Store) -> None:
    seed(ctx, store)
    body = _yaml(SEED / "templates" / "async_transfer.yaml")
    body["name"] = "odd_transfer"
    body["effects"]["transactions"][0]["kind"] = "warp_teleport"
    issues = validate_template(Template.model_validate(body), store)
    assert [(i.kind, i.severity) for i in issues if i.kind.startswith("unpriced")] == [
        ("unpriced_operation_class", "warning")
    ]


# ---- versions, regression set, interference ---------------------------------------------


def test_regression_set_needs_several_configurations_and_model_version_is_a_record(
    store: Store,
) -> None:
    one = {
        "name": "r",
        "version": 1,
        "target": "gb10",
        "cases": [{"kernel": "gemm_fp16_tilelang", "configurations": [{"BM": 128}]}],
    }
    with pytest.raises(ValueError, match="at least 2"):
        RegressionSet.model_validate(one)
    one["cases"][0]["configurations"].append({"BM": 64})  # type: ignore[index]
    rs = store.put(Record.new("RegressionSet", one, Actor(type="tool", name="t"), T0))
    mv = ModelVersion(
        target="gb10",
        label="seed",
        hw_model="gb10@1",
        arch="sm12x@1",
        simulators={"L1": "kta.models.l1@<commit>"},
        change="initial",
        reason="documented values only",
    )
    rid = store.put(
        Record.new(
            "ModelVersion", mv, Actor(type="tool", name="t"), T0, refs={"regression_set": rs}
        )
    )
    assert store.get(rid).body["admitted"] is None
    with pytest.raises(ValueError, match="change"):
        Record.new(
            "ModelVersion", {**mv.model_dump(), "change": "tweak"}, Actor(type="tool", name="t"), T0
        )


def test_interference_plan_defines_three_jobs_per_pair() -> None:
    classes = ["ffma", "mma_sync_m16n8k16_f16", "global_load"]
    jobs = interference.plan(classes)
    assert len(jobs) == 3 * 3
    ab = next(j for j in jobs if j.mode == "ab" and {j.a, j.b} == {"ffma", "global_load"})
    assert "fmaf" in ab.kernel.source and "__ldcg" in ab.kernel.source
    assert [p.name for p in ab.kernel.params][:2] == ["X", "n4"]
    m = interference.empty_matrix("gb10", "sm12x@1", classes)
    assert m.coverage() == {"pairs": 6, "measured": 0, "declared_independent": 0, "unmeasured": 6}
    c = interference.cell("ffma", "mufu", 1.0, 1.0, 1.9, ["x"])
    assert c.slowdown == pytest.approx(1.9)
    with pytest.raises(NotSupported, match="no interference probe"):
        interference.kernel("ffma", "warp_teleport", "a")


# ---- migration script --------------------------------------------------------------------


def _migration() -> Any:
    path = repo_root() / "scripts" / "migrations" / "WP15_effects.py"
    spec = importlib.util.spec_from_file_location("wp15_effects", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_is_idempotent_and_keeps_the_rest_of_the_file(tmp_path: Path) -> None:
    mig = _migration()
    rel = "templates/multi_buffer_rotation.yaml"
    path = tmp_path / "t.yaml"
    shutil.copy(SEED / rel, path)
    text = path.read_text(encoding="utf-8")
    start = text.index("effects:")
    end = text.index("scope:")
    old = 'cost:\n  - {unit: issue, work: "beta * p.iterations", overlap: serial}\n'
    path.write_text(text[:start] + old + text[end:], encoding="utf-8")
    assert mig.migrate(path, mig.EFFECTS[rel]) is True
    assert path.read_text(encoding="utf-8") == text
    assert mig.migrate(path, mig.EFFECTS[rel]) is False


def test_source_helper_still_usable(store: Store) -> None:
    # The seed batch's Source helper is shared with these tests; keep it importable.
    assert source_id(store)
    with pytest.raises(SchemaError):
        MachineModel.build(HwModel(target="toy", version=1, arch="other@1"), _toy()[1])
