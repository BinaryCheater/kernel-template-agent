"""The calibrate process (docs/design/11 §1): microbenchmarks, model refits, error alerts.

One tick:

1. runs the done-checks of submitted tasks booked to ``calibrate`` (T4, T5);
2. **hardware**: when the newest HwModel of the Target is missing or was measured on an
   older Target record (a Target change), measures the per-unit peaks
   (``kta.measure.peaks``) and writes a new HwModel; then runs every microbenchmark of
   ``kta.measure.microbench.SUITE`` whose HwFact is missing for this Target record. All
   GPU work goes through the Executor; a failure is recorded and retried after
   ``retry_s``;
3. **refit**: once ``refit_after`` new (Prediction, Measurement) pairs exist since the
   current model version was fitted, fits a new version on all pairs; it becomes current
   only when its hold-out error is below the current version's on the same hold-out
   points (else a ``calibration`` Report says why not) (D405);
4. **alerts**: per family, over the last ``window`` pairs, a median signed error beyond
   ``alert_rel`` creates one T4 (inputs: model, then Prediction/Measurement pairs, the
   layout ``model_holdout_improves`` reads), unless one is open for that family;
5. **T5**: every Gap with ``missing = hw_fact`` gets one T5 (input: the Gap; the check
   measures on the current Target).

Thresholds are in config ``[calibrate]`` (D405).
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from kta.core.config import Config
from kta.core.ids import record_id
from kta.core.registry import resolve
from kta.core.schemas import Actor, Record
from kta.core.schemas.evaluation import Measurement, Prediction
from kta.core.schemas.process import Report
from kta.measure.microbench import SUITE
from kta.measure.peaks import measure_peaks
from kta.models.calibrate import MODEL_SUBJECT, save_model
from kta.models.fit import errors
from kta.models.unit_model import UnitModel
from kta.orchestrator.proc_common import (
    ProcReport,
    action,
    check_submitted,
    get_state,
    guarded,
    live_tasks,
    process_of,
    update_state,
)
from kta.orchestrator.runner import Runner
from kta.orchestrator.transition import TaskRequest

ACTOR = Actor(type="system", name="kta.orchestrator.calibrate")
SUITE_FACT = {
    "l2_sweep": "l2_effective_bytes",
    "dram_latency": "dram_latency_s",
    "launch_latency": "launch_latency_s",
}


@dataclass(frozen=True)
class CalSettings:
    refit_after: int = 50
    window: int = 50
    alert_rel: float = 0.15
    min_pairs: int = 10
    retry_s: float = 3600.0

    @staticmethod
    def from_config(config: Config) -> CalSettings:
        sec = config.section("calibrate")
        d = CalSettings()
        return CalSettings(
            refit_after=int(sec.get("refit_after", d.refit_after)),
            window=int(sec.get("window", d.window)),
            alert_rel=float(sec.get("alert_rel", d.alert_rel)),
            min_pairs=int(sec.get("min_pairs", d.min_pairs)),
            retry_s=float(sec.get("retry_s", d.retry_s)),
        )


@dataclass(frozen=True)
class Pair:
    prediction: str
    measurement: str
    family: str
    target: str  # the Case's target: the unit model it is judged against


def solve_pairs(runner: Runner) -> list[Pair]:
    """(Prediction, Measurement, family) of every measured candidate of every solve loop."""
    store = runner.svc.store
    out: list[Pair] = []
    for row in runner.svc.ops.list_loops("solve"):
        d = row["data"]
        target = str(store.get(str(row["subject"])).body["target"])
        fams: list[Any] = list(d.get("families") or [])
        family = str(fams[0]) if fams else ""
        for cid, mid in sorted(dict(d.get("measurements") or {}).items()):
            preds = store.query("Prediction", {"refs.candidate": cid}, limit=1)
            if preds:
                out.append(Pair(record_id(preds[0]), str(mid), family, target))
    return out


def _load(runner: Runner, pairs: list[Pair]) -> list[tuple[Prediction, Measurement]]:
    get = runner.svc.store.get
    return [
        (
            Prediction.model_validate(get(p.prediction).body),
            Measurement.model_validate(get(p.measurement).body),
        )
        for p in pairs
    ]


# ---- hardware -------------------------------------------------------------------


def hardware(runner: Runner, rep: ProcReport) -> None:
    svc, ctx = runner.svc, runner.ctx
    plugin = resolve("target", ctx)
    target_id = str(plugin.latest_target_id())
    target = svc.target()
    protocol_id, _ = svc.protocol()
    limits = plugin.limits()
    m, cuda = svc.measurer(), svc.backend("cuda")
    found = svc.store.query("HwModel", {"target": target.name}, order="-created_at", limit=1)
    if not found or found[0].refs.get("target") != target_id:
        with action(runner, "calibrate", "peak microbenchmarks", target_id) as out:
            res = measure_peaks(
                m, cuda, target, target_id, limits, protocol_id, int(limits["sm_count"])
            )
            out["hwmodel"] = res.hwmodel_id
        rep.add("hwmodel", res.hwmodel_id)
    for name, fn in SUITE.items():
        have = svc.store.query(
            "HwFact", {"quantity": SUITE_FACT[name], "refs.target": target_id}, limit=1
        )
        if have:
            continue
        with action(runner, "calibrate", f"microbenchmark {name}", target_id) as out:
            res2 = fn(m, cuda, target, target_id, protocol_id)
            out["facts"] = sorted(res2.facts.values())
        rep.add("hwfacts", *res2.facts.values())


# ---- model ----------------------------------------------------------------------


def _median_err(model: UnitModel, data: list[tuple[Prediction, Measurement]]) -> float | None:
    e = errors(
        [model.total_us(p.per_unit, p.serial) for p, _ in data], [m.median_us for _, m in data]
    )
    v = e["median_abs_rel_err"]
    return None if v is None else float(v)


def refit(runner: Runner, rep: ProcReport, pairs: list[Pair], s: CalSettings) -> None:
    for target in sorted({p.target for p in pairs}):
        _refit(runner, rep, [p for p in pairs if p.target == target], s, target)


def _refit(runner: Runner, rep: ProcReport, pairs: list[Pair], s: CalSettings, target: str) -> None:
    svc = runner.svc
    model_id, model = svc.unit_model(target)
    fitted = {tuple(p) for p in svc.store.get(model_id).body["data"].get("fitted_on", [])}
    ids = [(p.prediction, p.measurement) for p in pairs]
    state = get_state(svc.ops, f"calibrate:fit:{target}")
    fresh = [x for x in ids if x not in fitted]
    if len(fresh) < s.refit_after or state.get("n_pairs") == len(ids):
        return
    data = _load(runner, pairs)
    with action(runner, "calibrate", "refit unit model", model_id, pairs=len(ids)) as out:
        new = model.fit(data)
        hold_idx: list[int] = [int(i) for i in new.fit_info.get("holdout_indices", [])]
        hold = [data[i] for i in hold_idx]
        old_err, new_err = _median_err(model, hold), _median_err(new, hold)
        out.update(old_error=old_err, new_error=new_err, holdout=len(hold))
        if hold and old_err is not None and new_err is not None and new_err < old_err:
            mid = save_model(runner.ctx, svc.store, new, ACTOR, previous=model_id, fitted_on=ids)
            out["model"] = mid
            rep.add("models", mid)
        else:
            body = Report(
                title="calibration",
                subject=MODEL_SUBJECT,
                summary=(
                    f"refit on {len(ids)} pairs not adopted: hold-out error {new_err} "
                    f"is not below {old_err} of {model_id[:12]}"
                ),
                data={"model": model_id, "old_error": old_err, "new_error": new_err, "n": len(ids)},
            )
            refs: dict[str, Any] = {
                "previous": model_id,
                "predictions": [p for p, _ in ids],
                "measurements": [m for _, m in ids],
            }
            rid = svc.store.put(Record.new("Report", body, ACTOR, runner.ctx.clock.now(), refs))
            rep.add("calibration_reports", rid)
    update_state(svc.ops, f"calibrate:fit:{target}", n_pairs=len(ids))


def alerts(runner: Runner, rep: ProcReport, pairs: list[Pair], s: CalSettings) -> None:
    svc = runner.svc
    open_for = {
        str(i.get("family")) for t, i in live_tasks(runner, "calibrate") if t.card_type == "T4"
    }
    by_family: dict[tuple[str, str], list[Pair]] = defaultdict(list)
    for p in pairs:
        by_family[(p.target, p.family)].append(p)
    for (target, family), items in sorted(by_family.items()):
        recent = items[-s.window :]
        if not family or family in open_for or len(recent) < s.min_pairs:
            continue
        model_id, model = svc.unit_model(target)
        data = _load(runner, recent)
        signed = [
            (model.total_us(p.per_unit, p.serial) - m.median_us) / m.median_us for p, m in data
        ]
        med = statistics.median(signed)
        if abs(med) <= s.alert_rel:
            continue
        name = str(svc.store.get(family).body.get("name", family[:12]))
        goal = (
            f"Diagnose the cost model: predictions for family {name} are "
            f"{'low' if med < 0 else 'high'} by a median {abs(med):.0%} over the last "
            f"{len(recent)} measurements"
        )
        inputs = [model_id] + [x for p in recent for x in (p.prediction, p.measurement)]
        tid = runner.queue.create(
            TaskRequest("T4", goal, inputs), f"calibrate:{family}", "calibrate"
        )
        update_state(svc.ops, f"task:{tid}", family=family, median_signed_error=med)
        rep.add("T4", tid)


def microbench_tasks(runner: Runner, rep: ProcReport) -> None:
    svc = runner.svc
    state = get_state(svc.ops, "calibrate:t5")
    handled = set(state.get("gaps", []))
    for gap in svc.store.query("Gap", {"missing": "hw_fact"}, order="created_at", limit=200):
        gid = record_id(gap)
        if gid in handled:
            continue
        need = gap.body["description"]
        goal = f"Write a microbenchmark for the hardware fact this Gap needs: {need}"
        tid = runner.queue.create(TaskRequest("T5", goal, [gid]), f"calibrate:{gid}", "calibrate")
        update_state(svc.ops, f"task:{tid}", gap=gid)
        handled.add(gid)
        rep.add("T5", tid)
    update_state(svc.ops, "calibrate:t5", gaps=sorted(handled))


def calibrate_tick(runner: Runner) -> ProcReport:
    rep = ProcReport("calibrate")
    s = CalSettings.from_config(runner.ctx.config)
    ops, now = runner.svc.ops, runner.ctx.clock.now()
    check_submitted(runner, rep, lambda t: process_of(runner, t) == "calibrate")
    guarded(rep, ops, "calibrate:hw:error", now, s.retry_s)(lambda: hardware(runner, rep))
    pairs = solve_pairs(runner)
    guarded(rep, ops, "calibrate:model:error", now, s.retry_s)(lambda: refit(runner, rep, pairs, s))
    guarded(rep, ops, "calibrate:alerts:error", now, s.retry_s)(
        lambda: alerts(runner, rep, pairs, s)
    )
    guarded(rep, ops, "calibrate:t5:error", now, s.retry_s)(lambda: microbench_tasks(runner, rep))
    return rep
