"""Human operations of the orchestrator, in the shape of the API's ``Orchestration``
(src/kta/api/wiring.py): task control, priorities, pausing and re-verification.

Each call writes one HumanAction naming the operator (D365) and returns a small dict for
the API. The profile pointer, batch verification and task tokens belong to the agents
module (layer 8, which the orchestrator may not import); ``missing`` names them so the
API can call ``kta.agents`` directly.
"""

from __future__ import annotations

import json
from typing import Any

from kta.core.errors import ConfigError, NotSupported
from kta.core.ids import record_id
from kta.core.schemas import Actor, Record
from kta.models import bound, min_work
from kta.orchestrator import control
from kta.orchestrator.machines import machine
from kta.orchestrator.pipeline import Services, bucket_shape, case_view
from kta.orchestrator.runner import PROCESS_OF, Runner
from kta.orchestrator.tasks import TaskQueue
from kta.orchestrator.transition import TaskRequest

ACTOR = Actor(type="system", name="kta.orchestrator.reverify")
AGENTS_SIDE = {
    "approve-profile": "kta.agents.profile.approve_profile (agents, layer 8)",
    "env-verify": "kta.agents.batches.verify (agents, layer 8)",
    "write-record": "kta.agents.guard.verify_task_token (agents, layer 8)",
}
# How a human advances a loop (WP25, D664): one step, until the round ends, or continuously
# (the loop's process takes over); ``retry`` clears a blocked mark and takes one step.
ADVANCE_MODES = ("step", "round", "run", "retry")
# "Advance one round" stops after this many steps even if the round has not ended: a guard
# against a loop that cycles without closing its round (not a research knob).
ROUND_STEP_CAP = 40

# Records a re-verification can act on: the Solution itself, or what a Solution rests on.
SOLUTION_REFS = ("measurement", "check", "bound")


def _action(svc: Services, human: Actor, action: str, reason: str, refs: dict[str, Any]) -> str:
    if human.type != "human":
        raise ConfigError(f"{action} needs a human actor")
    body = {"action": action, "reason": reason}
    return svc.store.put(Record.new("HumanAction", body, human, svc.ctx.clock.now(), refs=refs))


def _solutions_of(svc: Services, record: str) -> list[str]:
    rec = svc.store.get(record)
    if rec.kind == "Solution":
        return [record]
    found: list[str] = []
    for rel in SOLUTION_REFS:
        found += [record_id(r) for r in svc.store.query("Solution", {f"refs.{rel}": record})]
    return found


def reverify_bound(svc: Services, record: str) -> str:
    """Recompute a Bound with the newest HwModel and write the comparison (D378).

    The shape comes from the solve loop that wrote it (``data.bounds`` / ``data.buckets``).
    Returns the ``reverification`` Report ID; ``data.agrees`` says whether it reproduced.
    """
    old = svc.store.get(record)
    for row in svc.ops.list_loops("solve"):
        names = [k for k, v in dict(row["data"].get("bounds") or {}).items() if v == record]
        if not names:
            continue
        bucket = next(b for b in row["data"]["buckets"] if str(b["name"]) == names[0])
        cv = case_view(svc.store, str(row["subject"]))
        hw_id, hw = svc.machine(cv.case.target)
        shape = bucket_shape(bucket)
        new = bound(cv.case, shape, hw, min_work(cv.problem, shape))
        new_id = svc.store.put(
            Record.new(
                "Bound", new, ACTOR, svc.ctx.clock.now(), refs={"case": cv.id, "hw_model": hw_id}
            )
        )
        was = float(old.body["bound_us"])
        agrees = abs(new.bound_us - was) <= 1e-9 * max(1.0, abs(was))
        body = {
            "title": "reverification",
            "subject": record,
            "summary": f"Bound {'reproduced' if agrees else 'differs'}: "
            f"{was:.4g} us then, {new.bound_us:.4g} us now",
            "data": {
                "kind": "Bound",
                "old_us": was,
                "new_us": new.bound_us,
                "agrees": agrees,
                "bucket": names[0],
            },
        }
        return svc.store.put(
            Record.new(
                "Report", body, ACTOR, svc.ctx.clock.now(), refs={"subject": record, "new": new_id}
            )
        )
    raise NotSupported(f"Bound {record[:12]} was not written by a solve loop on this store")


def reverifiable(svc: Services, record: str) -> None:
    """NotSupported unless ``reverify`` can act on ``record`` (D368, D378)."""
    if not _solutions_of(svc, record) and svc.store.get(record).kind != "Bound":
        kind = svc.store.get(record).kind
        raise NotSupported(f"re-verifying a {kind} that no Solution rests on (WP11)")


def reverify(svc: Services, record: str, holder: str) -> list[str]:
    """Re-verify a suspect record (D368, D378); returns the loops started or the Report.

    A Solution, or a Measurement/Check/Bound a Solution rests on: a fresh solve-loop
    instance for its Case rebuilds, rechecks and remeasures. A Bound no Solution rests on
    is recomputed. Other kinds raise NotSupported.
    """
    solutions = _solutions_of(svc, record)
    if not solutions:
        kind = svc.store.get(record).kind
        if kind == "Bound":
            out = [reverify_bound(svc, record)]
            key = f"reverify:{record}"
            svc.ops.set_pointer(key, svc.ops.get_pointer(key), json.dumps({"reports": out}))
            return out
        raise NotSupported(f"re-verifying a {kind} that no Solution rests on (WP11)")
    runner = Runner(svc, holder)
    loops: list[str] = []
    for sid in solutions:
        case = svc.store.get(sid).refs["case"]
        assert isinstance(case, str)
        loops.append(runner.start("solve", case, f"re-verify {sid[:12]}"))
    key = f"reverify:{record}"
    svc.ops.set_pointer(key, svc.ops.get_pointer(key), json.dumps({"loops": loops}))
    return loops


def reverify_suspects(runner: Runner) -> list[str]:
    """Maintain process: every record marked suspect and not yet handled is re-verified;
    a kind that cannot be re-verified is marked so it is not retried every tick."""
    svc = runner.svc
    done: list[str] = []
    for change in svc.store.query("StatusChange", {"to_status": "suspect"}, limit=1000):
        subject = str(change.body["subject"])
        if svc.ops.get_pointer(f"reverify:{subject}") is not None:
            continue
        try:
            reverify(svc, subject, runner.holder)
        except NotSupported as exc:
            key = f"reverify:{subject}"
            svc.ops.set_pointer(key, None, json.dumps({"unsupported": str(exc)}))
            continue
        done.append(subject)
    return done


class Operations:
    def __init__(self, svc: Services) -> None:
        self.svc = svc
        self.queue = TaskQueue(svc.ctx, svc.store, svc.ops, svc)

    def missing(self) -> dict[str, str]:
        return dict(AGENTS_SIDE)

    def cancel_task(self, task_id: str, reason: str, human: Actor) -> dict[str, Any]:
        action = self.queue.cancel(task_id, reason, human)
        return {"task": task_id, "state": "cancelled", "action": action}

    def retry_task(self, task_id: str, note: str, human: Actor) -> dict[str, Any]:
        action = self.queue.retry(task_id, note, human, note=note)
        return {"task": task_id, "state": "queued", "action": action}

    def create_task(self, spec: Any, reason: str, human: Actor) -> dict[str, Any]:
        """``spec`` has ``type``, ``goal``, ``inputs`` and ``priority`` (api TaskSpec)."""
        req = TaskRequest(str(spec.type), str(spec.goal), list(spec.inputs))
        task = self.queue.create(req, None, "human", int(spec.priority), human, reason)
        return {"task": task, "work_unit": self.queue.get(task).work_unit, "state": "queued"}

    def set_priority(
        self, task_id: str, priority: int, reason: str, human: Actor
    ) -> dict[str, Any]:
        action = self.queue.set_priority(task_id, priority, reason, human)
        return {"task": task_id, "priority": priority, "action": action}

    def _pause(self, scope: str, paused: bool, reason: str, human: Actor) -> dict[str, Any]:
        verb = "pause" if paused else "resume"
        action = _action(self.svc, human, f"{verb} {scope}", reason, {})
        control.set_paused(
            self.svc.ops, scope, paused, reason, human.name, self.svc.ctx.clock.now()
        )
        return {"scope": scope, "paused": paused, "action": action}

    def pause(self, scope: str, reason: str, human: Actor) -> dict[str, Any]:
        return self._pause(scope, True, reason, human)

    def resume(self, scope: str, reason: str, human: Actor) -> dict[str, Any]:
        return self._pause(scope, False, reason, human)

    def advance_action(self, loop_id: str, mode: str, reason: str, human: Actor) -> dict[str, Any]:
        """Check and record a human's advance request; the work is ``advance_work``.

        ``run`` resumes the loop so its process advances it; the reply says whether that
        process is running (``process_running``), since nothing moves otherwise."""
        if mode not in ADVANCE_MODES:
            raise ConfigError(f"advance mode {mode!r}; one of {', '.join(ADVANCE_MODES)}")
        row = self.svc.ops.get_loop(loop_id)
        params = json.dumps({"loop": loop_id, "mode": mode}, sort_keys=True)
        action = _action(self.svc, human, f"advance {params}", reason, {"subject": row["subject"]})
        out: dict[str, Any] = {"loop": loop_id, "mode": mode, "action": action}
        if mode == "retry" and row["data"].get("blocked"):
            data = {**row["data"], "blocked": None, "infra_failures": 0}
            self.svc.ops.put_loop(loop_id, row["kind"], row["subject"], row["state"], data)
        if mode == "run":
            scope = f"loop:{loop_id}"
            if control.is_paused(self.svc.ops, scope):
                control.set_paused(
                    self.svc.ops, scope, False, reason, human.name, self.svc.ctx.clock.now()
                )
            process = PROCESS_OF[row["kind"]]
            out["process"] = process
            out["process_running"] = self.svc.ops.lock_holder(f"process:{process}") is not None
        return out

    def advance_work(self, loop_id: str, mode: str, holder: str) -> list[dict[str, Any]]:
        """Take the steps of ``advance_action``'s request (``step``/``retry``: one;
        ``round``: until the round number changes, the loop waits, ends, blocks or does not
        move). Returns the transitions taken."""
        if mode == "run":
            return []
        runner = Runner(self.svc, holder)
        start_round = self.svc.ops.get_loop(loop_id)["data"].get("round")
        taken: list[dict[str, Any]] = []
        for _ in range(1 if mode in ("step", "retry") else ROUND_STEP_CAP):
            t = runner.advance(loop_id, manual=True)
            if t is None:
                break
            taken.append(
                {"from": t.src, "to": t.to, "kind": t.kind, "rule": t.rule, "note": t.note}
            )
            row = self.svc.ops.get_loop(loop_id)
            if (
                row["data"].get("round") != start_round
                or row["data"].get("waiting_on")
                or row["data"].get("blocked")
                or machine(row["kind"]).terminal(row["state"])
            ):
                break
        return taken

    def process_action(self, name: str, start: bool, reason: str, human: Actor) -> dict[str, Any]:
        """Record a human's start or stop of a process (WP25, D668). A stop is a stop
        request the running process reads at its next tick; a start is checked here and
        run by the caller (``processes.run`` holds the process lock while it runs)."""
        from kta.orchestrator import processes  # processes imports this module

        if name not in processes.ALL:
            raise ConfigError(f"unknown process {name!r}; one of {', '.join(processes.ALL)}")
        holder = self.svc.ops.lock_holder(f"process:{name}")
        if start and holder is not None:
            raise ConfigError(f"process {name} is already running ({holder})")
        verb = "start" if start else "stop"
        action = _action(self.svc, human, f"process {verb} {name}", reason, {})
        if not start:
            processes.request_stop(self.svc, name)
        return {"process": name, "action": action, "running": holder is not None}

    def run_process(self, name: str, interval_s: float) -> int:
        """Run ``name`` until a stop request (blocks; the API calls it on a thread)."""
        from kta.orchestrator import processes  # processes imports this module

        return processes.run(self.svc, name, interval_s)

    def request_reverify(self, record: str, reason: str, human: Actor) -> dict[str, Any]:
        reverifiable(self.svc, record)  # refuse before writing the HumanAction
        action = _action(self.svc, human, f"reverify {record}", reason, {"subject": record})
        loops = reverify(self.svc, record, f"human:{human.name}")
        return {"record": record, "loops": loops, "action": action}

    def _agents(self, name: str) -> NotSupported:
        return NotSupported(f"{name}: call {AGENTS_SIDE[name]} directly")

    def approve_profile(self, profile: str, reason: str, human: Actor) -> dict[str, Any]:
        raise self._agents("approve-profile")

    def verify_batch(self, batch: str) -> dict[str, Any]:
        raise self._agents("env-verify")

    def verify_task_token(self, token: str, now: Any) -> str:
        raise self._agents("write-record")
