"""``run_task``: one agent step of a leased task, from task card to AgentRun record.

Order: card and profile → session decision → step bundle → task token and whitelisted
environment → launcher job on the agent host (through the Executor, D300) → session
events → compliance → AgentRun → session state. The runner does not move the task
between ``running``/``submitted``/``accepted`` states: ``kta submit`` and the done checks
(orchestrator, WP6) do. It renews the task lease while the job runs.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from kta.agents import cards, compliance, guard, sessions, workspace
from kta.agents.adapters.base import StepPlan
from kta.agents.adapters.codex import CREDENTIAL_FILES, CodexAdapter
from kta.agents.adapters.codex_session import usage_of
from kta.agents.base import AgentEvent, Launch, Limits
from kta.agents.config import AgentsConfig, agents_config
from kta.agents.launchspec import BUNDLE_HOME, BUNDLE_WORKSPACE, AuthSpec, LaunchSpec
from kta.core.context import Context
from kta.core.errors import ConfigError, NotSupported
from kta.core.ids import UlidFactory, sha256_hex
from kta.core.jobs import JobRunner
from kta.core.registry import resolve
from kta.core.schemas import Actor, Record
from kta.core.schemas.process import AgentProfile, AgentRun, TaskCard
from kta.core.trace import child_environ, span
from kta.store.base import Ops, Store, Task

RENEW_EVERY_S = 300
LEASE_S = 1800  # docs/design/10 §4.3 default lease


@dataclass(frozen=True)
class Deps:
    store: Store
    ops: Ops
    runner: JobRunner
    config: AgentsConfig


@dataclass(frozen=True)
class StepResult:
    run_id: str  # AgentRun record ID
    run: AgentRun
    launch: Launch
    events: list[AgentEvent] = field(default_factory=lambda: list[AgentEvent]())
    decision: str = ""


def deps_from(ctx: Context) -> Deps:
    return Deps(
        resolve("store", ctx), resolve("ops", ctx), resolve("executor", ctx), agents_config(ctx)
    )


def adapter_for(profile: AgentProfile, deps: Deps, plan: StepPlan) -> CodexAdapter:
    if profile.provider != "codex":
        raise NotSupported(
            f"provider {profile.provider}: only the codex adapter is implemented (WP8)"
        )
    return CodexAdapter(
        deps.runner,
        plan,
        deps.config.host.codex_cmd,
        profile.cli_flags,
        profile.model,
        profile.reasoning_effort,
    )


def _put_files(runner: JobRunner, prefix: str, files: dict[str, bytes]) -> dict[str, str]:
    return {
        prefix + name: runner.put_blob(data, "application/octet-stream")
        for name, data in files.items()
    }


def _limits(card: TaskCard, profile: AgentProfile) -> Limits:
    loop = profile.loop_policy
    wall = card.budget.get("wall_s") or dict(loop.get("wall_s", {})).get(card.type)
    if not wall:
        raise ConfigError(f"no wall budget for {card.type} in the card or the profile")
    return Limits(wall_s=int(wall), idle_s=int(loop.get("idle_s", 600)))


def _env(ctx: Context, deps: Deps, token: str, task: Task) -> dict[str, str]:
    values = {
        guard.TOKEN_VAR: token,
        "KTA_TASK": task.id,
        "KTA_ENV": ctx.env,
        "HOME": "{home}",
        "CODEX_HOME": "{home}",
        "LANG": "C.UTF-8",
        **child_environ(),
    }
    if deps.config.api_url:
        values["KTA_API"] = deps.config.api_url
    if ctx.env == "dev" and os.environ.get("KTA_WP"):
        values["KTA_WP"] = os.environ["KTA_WP"]
    return guard.agent_env(values, ["{kta_bin}"])


def _spec(
    ctx: Context,
    deps: Deps,
    task: Task,
    card: TaskCard,
    profile: AgentProfile,
    run: str,
    bundle: workspace.Bundle,
) -> LaunchSpec:
    run_as = None
    if profile.isolation == "config+user":
        if not deps.config.agent_user:
            raise ConfigError(
                "profile isolation config+user needs [agents] agent_user (scripts/kta-agent-setup)"
            )
        run_as = deps.config.agent_user
    elif profile.isolation == "container":
        raise NotSupported("container isolation (D09) is not implemented")
    host = deps.config.host
    auth = (
        AuthSpec("codex", host.auth_dir, list(CREDENTIAL_FILES))
        if profile.auth == "subscription"
        else None
    )
    return LaunchSpec(
        provider=profile.provider,
        work_unit=card.work_unit,
        run=run,
        mode="start",
        workspace_files=sorted(bundle.workspace),
        home_files=sorted(bundle.home),
        auth=auth,
        run_as=run_as,
        secret_env=[guard.TOKEN_VAR],
        notes_file="NOTES.md",
        nvidia_smi=host.nvidia_smi,
        watch=list(deps.config.watch),
        expect_version=profile.cli_version,
    )


def _renewing(ops: Ops, task: Task, stop: threading.Event) -> None:
    owner = task.lease_owner or "agent-runner"
    while not stop.wait(RENEW_EVERY_S):
        ops.renew(task.id, owner, LEASE_S)


def _tar_hashes(data: bytes) -> dict[str, str]:
    out: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar.getmembers():
            handle = tar.extractfile(m) if m.isfile() else None
            if handle is not None:
                out[m.name] = sha256_hex(handle.read())
    return out


def tool_seconds(events: list[AgentEvent]) -> float:
    """Time spent inside ``kta`` commands, from the exec results of the session."""
    total = 0.0
    for ev in events:
        exec_ = ev.data.get("exec") or {}
        words = " ".join(str(a) for a in exec_.get("command") or [])
        if " kta " in f" {words} " or words.startswith("kta "):
            total += float(exec_.get("duration_s") or 0.0)
    return total


def violations_of(
    deps: Deps, launch: Launch, events: list[AgentEvent], profile: AgentProfile, card: TaskCard
) -> list[dict[str, Any]]:
    allowed = profile.allowed_commands.get(card.type, [])
    report = launch.report
    out = compliance.from_events(events, str(report.get("workspace", "")), allowed)
    out += compliance.from_report(report, bool(profile.context_policy.get("notes_required", True)))
    if launch.manifest and launch.workspace:
        mf = json.loads(deps.runner.read_blob(launch.manifest))
        after = _tar_hashes(deps.runner.read_blob(launch.workspace))
        out += compliance.context_changes(mf["workspace"]["files"], after)
    return out


def _prompt(
    store: Store,
    profile: AgentProfile,
    mode: str,
    card: TaskCard,
    task: Task,
    retry: cards.RetryNote | None,
    note: str | None,
) -> str:
    def tpl(name: str) -> str:
        return store.blob_path(profile.message_templates[name]).read_text(encoding="utf-8")

    parts: list[str] = []
    if mode == "start":
        parts.append(cards.new_session(tpl("new_session"), card.work_unit, retry))
    elif retry is not None:
        parts.append(cards.rejection(tpl("rejection"), task.id, retry))
    if note:
        parts.append(cards.human_note(tpl("human_note"), note))
    if not parts:
        parts.append("Continue with TASK.md (updated for this step); read NOTES.md first.")
    return "\n\n".join(parts)


def run_task(
    ctx: Context,
    task: Task,
    *,
    retry: cards.RetryNote | None = None,
    note: str | None = None,
    deps: Deps | None = None,
    lane: str = "agent:0",
) -> StepResult:
    """Run one step of ``task`` and write its AgentRun (spec ``run_task(ctx, task)``).

    ``retry`` carries the rejection of the previous attempt (from the orchestrator);
    ``note`` is a human's message, passed verbatim.
    """
    deps = deps or deps_from(ctx)
    card = TaskCard.model_validate(deps.store.get(task.card_id).body)
    profile = AgentProfile.model_validate(deps.store.get(card.profile).body)
    run = UlidFactory(ctx.clock).new()
    state, raw = sessions.load(deps.ops, card.work_unit)
    if state is not None and state.worker and state.worker != deps.config.worker:
        raise ConfigError(
            f"work unit {card.work_unit} lives on {state.worker}, not {deps.config.worker}"
        )
    bundle = workspace.build(
        deps.store,
        profile,
        task_id=task.id,
        card_id=task.card_id,
        card_body=card,
        used={},
        retry=retry,
    )
    limits = _limits(card, profile)
    spec = _spec(ctx, deps, task, card, profile, run, bundle)
    inputs = {
        **_put_files(deps.runner, BUNDLE_WORKSPACE, bundle.workspace),
        **_put_files(deps.runner, BUNDLE_HOME, bundle.home),
    }
    plan = StepPlan(spec, inputs, {"pool": deps.config.pool}, hard_cap_s=3 * limits.wall_s)
    adapter = adapter_for(profile, deps, plan)
    decision = sessions.decide(
        state, adapter.supports_resume, profile.loop_policy, retry.signature if retry else None
    )
    key = guard.ensure_key(Path(deps.config.token_key_file))
    token = guard.issue_token(key, task.id, ctx.clock.now() + timedelta(seconds=plan.hard_cap_s))
    prompt = _prompt(deps.store, profile, decision.mode, card, task, retry, note)
    stop = threading.Event()
    renewer = threading.Thread(target=_renewing, args=(deps.ops, task, stop), daemon=True)
    with span(
        "agent_session",
        f"{card.type} {card.work_unit}",
        lane,
        task_id=task.id,
        run=run,
        mode=decision.mode,
    ) as sp:
        env = _env(ctx, deps, token, task)
        renewer.start()
        try:
            events = _launch(adapter, decision, state, prompt, card, env, limits)
        finally:
            stop.set()
        launch = adapter.last_launch()
        assert launch is not None
        sp.set(session=launch.session, exit=launch.exit, worker=launch.worker)
    return _record(
        ctx, deps, task, card, profile, launch, events, prompt, state, raw, decision, retry
    )


def _launch(
    adapter: CodexAdapter,
    decision: sessions.Decision,
    state: sessions.SessionState | None,
    prompt: str,
    card: TaskCard,
    env: dict[str, str],
    limits: Limits,
) -> list[AgentEvent]:
    ws = Path("work") / card.work_unit
    if decision.mode == "resume" and state is not None and state.session:
        it: Iterator[AgentEvent] = adapter.resume(state.session, prompt, ws, env, limits)
        return list(it)
    _, it = adapter.start(prompt, ws, env, limits)
    return list(it)


def _record(
    ctx: Context,
    deps: Deps,
    task: Task,
    card: TaskCard,
    profile: AgentProfile,
    launch: Launch,
    events: list[AgentEvent],
    prompt: str,
    state: sessions.SessionState | None,
    raw: str | None,
    decision: sessions.Decision,
    retry: cards.RetryNote | None,
) -> StepResult:
    usage = usage_of(events)
    violations = violations_of(deps, launch, events, profile, card)
    body = AgentRun(
        session=launch.session,
        profile=card.profile,
        cli_version=str(launch.report.get("cli_version", "")),
        model=profile.model,
        tokens_in=usage.tokens_in,
        tokens_out=usage.tokens_out,
        session_archive=launch.archive,
        wall_s=launch.wall_s,
        tool_s=tool_seconds(events),
        queue_s=launch.queue_s,
        exit=launch.exit,
        context_manifest=launch.manifest,
        violations=violations,
    )
    refs: dict[str, Any] = {
        "env": launch.env_manifest,
        "worker": launch.worker,
        "card": task.card_id,
        "profile": card.profile,
    }
    if state is not None and state.runs:
        refs["previous"] = state.runs[-1]
    blobs = {"prompt": deps.runner.put_blob(prompt.encode("utf-8"), "text/markdown")}
    blobs.update(
        {k: v for k, v in (("stdout", launch.stdout), ("workspace", launch.workspace)) if v}
    )
    rec = Record.new(
        "AgentRun",
        body,
        Actor(type="system", name="kta.agents.runner", task=task.id),
        ctx.clock.now(),
        refs=refs,
        blobs=blobs,
    )
    run_id = deps.store.put(rec)
    new_state = sessions.after_run(
        state,
        session=launch.session,
        worker=deps.config.worker,
        run_id=run_id,
        exit=launch.exit,
        context_ratio=usage.context_ratio,
        failure=retry.signature if retry else None,
        started_new=decision.mode == "start",
    )
    sessions.save(deps.ops, card.work_unit, raw, new_state)
    return StepResult(run_id, body, launch, events, f"{decision.mode}: {decision.reason}")


def parse_session(path: Path) -> list[AgentEvent]:
    """Events of a saved Codex session file, CODEX_HOME directory or archive."""
    from kta.agents.adapters.codex import parse_session as codex_parse

    return codex_parse(path)


__all__ = ["Deps", "StepResult", "parse_session", "run_task"]
