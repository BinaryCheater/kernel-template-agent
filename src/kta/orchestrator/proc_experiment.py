"""The experiment process (docs/design/11 §1, docs/design/50 §1.5): approved protocols only.

An experiment is a Protocol record (``type = experiment``, D345) that you approved with a
HumanAction ``approve_protocol`` (``refs.protocol``). A Protocol record is immutable, so
approving it freezes it; changing a protocol means a new version and a new approval.
Its ``params`` (D403):

- ``arms`` (names), ``runs_per_arm``, ``cases`` (Case IDs): one run per arm x case x rep;
- ``snapshot`` (PoolSnapshot ID) and ``profile`` (AgentProfile ID): fixed for every run;
- ``budget_per_run`` (``agent_min``, ``gpu_min``): the same for every arm (50 §1.4);
- optional ``conditions`` {arm: {...}} (the knowledge condition), ``batch`` (batch
  environment name, default ``exp-<name>-v<version>``), ``max_concurrent`` (default 2).

One tick per approved, unfinished protocol: build (if absent) and verify its batch
environment through the ``batch_envs`` interface (``kta agents env build/verify``,
D322); start runs up to ``max_concurrent``, each a distinct solve loop booked to this
process (loop data ``process = experiment``, ``experiment`` = the run's pinned settings,
loop budget = ``budget_per_run``); tag each task with the batch; check and advance its
loops; write one ExperimentRun per finished run; when all runs are written, write the
result Report (``title = experiment``, ``data.rows``). Every run is kept and reported,
failed ones included (50 §1.5). The set of running protocols moves the process quotas
(``quotas.apply_quotas``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, cast

from kta.core.errors import SchemaError
from kta.core.registry import resolve
from kta.core.schemas import Actor, Record
from kta.core.schemas.process import ExperimentRun, Report
from kta.orchestrator import quotas
from kta.orchestrator.machines import machine
from kta.orchestrator.proc_common import (
    ProcReport,
    action,
    check_submitted,
    get_state,
    guarded,
    put_state,
    update_state,
)
from kta.orchestrator.runner import Runner
from kta.store.base import Store

ACTOR = Actor(type="system", name="kta.orchestrator.experiment")
APPROVE = "approve_protocol"
RETRY_S = 600.0
# Recorded with every run until solve-loop retrieval takes the pinned snapshot and the
# arm's knowledge condition (WP11 known issue; needed change for WP6/WP8/knowledge).
PINNING_NOTE = (
    "snapshot, profile, batch and arm condition are pinned in the loop data; solve-loop "
    "retrieval does not read them yet"
)


@dataclass(frozen=True)
class ExperimentSpec:
    protocol: str
    name: str
    version: int
    arms: list[str]
    runs_per_arm: int
    cases: list[str]
    snapshot: str
    profile: str
    budget_per_run: dict[str, float]
    conditions: dict[str, Any]
    batch: str
    max_concurrent: int

    @staticmethod
    def of(store: Store, protocol_id: str) -> ExperimentSpec:
        rec = store.get(protocol_id)
        body = rec.body
        if rec.kind != "Protocol" or body.get("type") != "experiment":
            raise SchemaError(f"{protocol_id} is not an experiment Protocol")
        p = dict(body.get("params", {}))
        need = ("arms", "runs_per_arm", "cases", "snapshot", "profile", "budget_per_run")
        missing = [k for k in need if not p.get(k)]
        if missing:
            raise SchemaError(f"experiment {body['name']}: params miss {missing} (D403)")
        for rid, kind in [(p["snapshot"], "PoolSnapshot"), (p["profile"], "AgentProfile")]:
            if store.get(str(rid)).kind != kind:
                raise SchemaError(f"experiment {body['name']}: {rid} is not a {kind}")
        for c in p["cases"]:
            if store.get(str(c)).kind != "Case":
                raise SchemaError(f"experiment {body['name']}: {c} is not a Case")
        arms = [str(a) for a in p["arms"]]
        if len(set(arms)) != len(arms):
            raise SchemaError(f"experiment {body['name']}: arm names repeat")
        return ExperimentSpec(
            protocol=protocol_id,
            name=str(body["name"]),
            version=int(body["version"]),
            arms=arms,
            runs_per_arm=int(p["runs_per_arm"]),
            cases=[str(c) for c in p["cases"]],
            snapshot=str(p["snapshot"]),
            profile=str(p["profile"]),
            budget_per_run={k: float(v) for k, v in dict(p["budget_per_run"]).items()},
            conditions=dict(p.get("conditions", {})),
            batch=str(p.get("batch") or f"exp-{body['name']}-v{body['version']}"),
            max_concurrent=int(p.get("max_concurrent", 2)),
        )

    def runs(self) -> list[tuple[str, str, int]]:
        """(arm, case, rep), interleaving arms so a budget cut leaves them balanced."""
        return [
            (arm, case, rep)
            for rep in range(self.runs_per_arm)
            for case in self.cases
            for arm in self.arms
        ]


def run_key(arm: str, case: str, rep: int) -> str:
    return f"{arm}/{case}/{rep}"


def approved(store: Store) -> list[str]:
    """Experiment Protocol IDs with an approval, in approval order."""
    out: list[str] = []
    for rec in store.query("HumanAction", None, order="created_at", limit=10000):
        pid = rec.refs.get("protocol")
        if str(rec.body.get("action", "")).startswith(APPROVE) and isinstance(pid, str):
            proto = store.get(pid)
            if proto.kind == "Protocol" and proto.body.get("type") == "experiment":
                if pid not in out:
                    out.append(pid)
    return out


def _finished(row: dict[str, Any]) -> bool:
    return machine("solve").terminal(str(row["state"])) or bool(row["data"].get("escalated"))


def _write_run(runner: Runner, spec: ExperimentSpec, key: str, loop_id: str) -> str:
    svc = runner.svc
    arm, case, rep = key.split("/")[0], key.split("/")[1], int(key.split("/")[2])
    row = svc.ops.get_loop(loop_id)
    d = row["data"]
    judgement = d.get("judgement")
    solutions: list[str] = []
    if judgement:
        sol = svc.store.get(str(judgement)).refs.get("solution")
        solutions = [sol] if isinstance(sol, str) else list(sol or [])
    used = svc.ops.budget(f"loop:{loop_id}")
    params = {
        "case": case,
        "loop": loop_id,
        "state": row["state"],
        "escalated": bool(d.get("escalated")),
        "rounds": d.get("round"),
        "judgement": judgement,
        "solutions": solutions,
        "snapshot": spec.snapshot,
        "profile": spec.profile,
        "batch": spec.batch,
        "arm_condition": spec.conditions.get(arm, {}),
        "budget": used,
    }
    body = ExperimentRun(name=spec.name, condition=arm, seed=rep, params=params, note=PINNING_NOTE)
    refs: dict[str, Any] = {
        "protocol": spec.protocol,
        "case": case,
        "snapshot": spec.snapshot,
        "profile": spec.profile,
    }
    if judgement:
        refs["judgement"] = str(judgement)
    if solutions:
        refs["solutions"] = solutions
    return svc.store.put(Record.new("ExperimentRun", body, ACTOR, runner.ctx.clock.now(), refs))


def _write_result(runner: Runner, spec: ExperimentSpec, runs: dict[str, str]) -> str:
    store = runner.svc.store
    rows: list[dict[str, Any]] = []
    for _, rid in sorted(runs.items()):
        run = store.get(rid).body
        pct = [float(store.get(s).body["pct_of_bound"]) for s in run["params"].get("solutions", [])]
        rows.append(
            {
                "arm": run["condition"],
                "case": run["params"]["case"],
                "rep": run["seed"],
                "run": rid,
                "state": run["params"]["state"],
                "escalated": run["params"]["escalated"],
                "pct_of_bound": min(pct) if pct else None,
                "used": run["params"]["budget"].get("used", {}),
            }
        )
    body = Report(
        title="experiment",
        subject=spec.protocol,
        summary=f"{spec.name} v{spec.version}: {len(rows)} runs over {len(spec.arms)} arms",
        data={"rows": rows, "final": True, "note": PINNING_NOTE},
    )
    refs = {"protocol": spec.protocol, "runs": sorted(runs.values())}
    return store.put(Record.new("Report", body, ACTOR, runner.ctx.clock.now(), refs))


def _tag_tasks(runner: Runner, loops: set[str], batch: str) -> None:
    for t in runner.svc.ops.list_tasks():
        if t.work_unit in loops and "batch" not in runner.queue.info(t.id):
            update_state(runner.svc.ops, f"task:{t.id}", batch=batch)


def advance_protocol(runner: Runner, rep: ProcReport, pid: str) -> None:
    svc = runner.svc
    key = f"experiment:{pid}"
    spec = ExperimentSpec.of(svc.store, pid)
    st = get_state(svc.ops, key)
    if not st:
        with action(runner, "experiment", f"batch env {spec.batch}", pid) as out:
            env = resolve("batch_envs", runner.ctx).ensure(spec.batch, spec.profile)
            out.update(env)
        st: dict[str, Any] = {"status": "running", "env": env, "slots": {}}
        put_state(svc.ops, key, st)
        rep.add("environments", spec.batch)
    else:  # verified before every tick's runs (D322)
        resolve("batch_envs", runner.ctx).ensure(spec.batch, spec.profile)
    slots = cast(dict[str, dict[str, Any]], st["slots"])
    loops = {s["loop"] for s in slots.values()}
    check_submitted(runner, rep, lambda t: t.work_unit in loops)
    _tag_tasks(runner, loops, spec.batch)
    for k, s in sorted(slots.items()):
        if s.get("run"):
            continue
        if _finished(svc.ops.get_loop(s["loop"])):
            s["run"] = _write_run(runner, spec, k, s["loop"])
            rep.add("runs", s["run"])
        elif runner.advance(s["loop"]) is not None:
            rep.add("advanced", s["loop"])
    live = sum(1 for s in slots.values() if not s.get("run"))
    for arm, case, i in spec.runs():
        k = run_key(arm, case, i)
        if k in slots or live >= spec.max_concurrent:
            continue
        pinned = {
            "protocol": pid,
            "arm": arm,
            "rep": i,
            "snapshot": spec.snapshot,
            "profile": spec.profile,
            "batch": spec.batch,
            "condition": spec.conditions.get(arm, {}),
        }
        loop = runner.start(
            "solve",
            case,
            f"{spec.name} {arm} #{i}",
            extra={"process": "experiment", "experiment": pinned},
            distinct=True,
        )
        svc.ops.set_budget(f"loop:{loop}", spec.budget_per_run)
        slots[k] = {"loop": loop}
        live += 1
        rep.add("started", loop)
    runs = {k: str(s["run"]) for k, s in slots.items() if s.get("run")}
    if len(runs) == len(spec.runs()):
        st["report"] = _write_result(runner, spec, runs)
        st["status"] = "done"
        rep.add("results", st["report"])
    put_state(svc.ops, key, st)


def experiment_tick(runner: Runner) -> ProcReport:
    rep = ProcReport("experiment")
    svc, now = runner.svc, runner.ctx.clock.now()
    running: list[str] = []
    for pid in approved(svc.store):
        if get_state(svc.ops, f"experiment:{pid}").get("status") == "done":
            continue
        guarded(rep, svc.ops, f"experiment:{pid}:error", now, RETRY_S)(
            partial(advance_protocol, runner, rep, pid)
        )
        if get_state(svc.ops, f"experiment:{pid}").get("status") == "running":
            running.append(pid)
    if quotas.set_active_experiments(svc.ops, running):
        shares = quotas.apply_quotas(svc.ops, runner.ctx.config)
        rep.add("quotas", *(f"{k}={v:.3f}" for k, v in sorted(shares.items())))
    return rep
