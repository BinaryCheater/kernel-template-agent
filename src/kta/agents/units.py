"""End of a work unit: archive the workspace and CLI home, then remove them.

Two launcher jobs (conventions §7.1: no directory is deleted before its archive is
stored): ``close`` packs both directories and returns them by hash; the runner stores
them and writes a Report (subject ``work_unit_archive``); only then ``remove`` repacks,
checks that the bytes are unchanged and deletes the directories.
"""

from __future__ import annotations

import json
from dataclasses import replace

from kta.agents import sessions
from kta.agents.adapters.base import StepPlan, launch_job
from kta.agents.adapters.codex import CREDENTIAL_FILES
from kta.agents.config import AgentsConfig
from kta.agents.launchspec import OUT_ARCHIVE, OUT_REPORT, OUT_WORKSPACE, AuthSpec, LaunchSpec
from kta.core.context import Context
from kta.core.errors import ConfigError, ExecutionError
from kta.core.ids import UlidFactory
from kta.core.jobs import JobResult, JobRunner
from kta.core.schemas import Actor, Record
from kta.core.schemas.process import Report
from kta.store.base import Ops, Store

CLOSE_TIMEOUT_S = 1800


def _report(runner: JobRunner, result: JobResult) -> dict[str, object]:
    if not result.ok or OUT_REPORT not in result.outputs:
        raise ExecutionError("infra", f"work-unit job {result.job_id} failed ({result.error_kind})")
    return json.loads(runner.read_blob(result.outputs[OUT_REPORT]))


def close_work_unit(
    ctx: Context, store: Store, ops: Ops, runner: JobRunner, config: AgentsConfig, work_unit: str
) -> str:
    """Archive and remove a work unit's directories; returns the archive Report ID."""
    state, raw = sessions.load(ops, work_unit)
    if state is None:
        raise ConfigError(f"work unit {work_unit} has no agent session state")
    auth = AuthSpec("codex", config.host.auth_dir, list(CREDENTIAL_FILES))
    spec = LaunchSpec("codex", work_unit, UlidFactory(ctx.clock).new(), "close", auth=auth)
    plan = StepPlan(spec, {}, {"pool": config.pool}, hard_cap_s=CLOSE_TIMEOUT_S)
    closed = launch_job(runner, plan, spec, "")
    _report(runner, closed)
    home, ws = closed.outputs[OUT_ARCHIVE], closed.outputs[OUT_WORKSPACE]
    rec = Record.new(
        "Report",
        Report(
            title=f"work unit {work_unit} archive",
            subject="work_unit_archive",
            summary=f"workspace with git history and CLI home of {work_unit}, no credentials",
            data={"work_unit": work_unit, "session": state.session, "runs": state.runs},
        ),
        Actor(type="system", name="kta.agents.units"),
        ctx.clock.now(),
        refs={"env": closed.env_manifest, "runs": list(state.runs)},
        blobs={"home": home, "workspace": ws},
    )
    report_id = store.put(rec)
    confirm = runner.put_blob(
        json.dumps({"home": home, "workspace": ws}).encode(), "application/json"
    )
    remove = replace(spec, mode="remove", run=UlidFactory(ctx.clock).new())
    removed = _report(
        runner, launch_job(runner, replace(plan, inputs={"confirm.json": confirm}), remove, "")
    )
    if removed.get("exit") != "ok":
        raise ExecutionError("infra", f"work unit {work_unit} not removed: {removed.get('error')}")
    sessions.save(ops, work_unit, raw, sessions.closed(state))
    return report_id
