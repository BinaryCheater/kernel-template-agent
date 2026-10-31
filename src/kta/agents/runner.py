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

from kta.agents import (
    arms,
    batches,
    cards,
    compliance,
    guard,
    policy,
    proxyplan,
    sessions,
    workspace,
)
from kta.agents.adapters.base import StepPlan
from kta.agents.adapters.codex import CREDENTIAL_FILES, CodexAdapter
from kta.agents.adapters.codex_session import usage_of
from kta.agents.arms import Arm
from kta.agents.base import AgentEvent, Launch, Limits
from kta.agents.batches import BatchRoot
from kta.agents.config import AgentsConfig, agents_config
from kta.agents.launchspec import BUNDLE_BATCH, BUNDLE_WORKSPACE, AuthSpec, LaunchSpec
from kta.agents.proxyplan import ProxyPlan
from kta.core.context import Context
from kta.core.controls import Controls
from kta.core.errors import ConfigError, NotSupported
from kta.core.ids import UlidFactory, sha256_hex
from kta.core.jobs import JobRunner
from kta.core.registry import resolve
from kta.core.schemas import Actor, Record
from kta.core.schemas.process import AgentProfile, AgentRun, TaskCard
from kta.core.trace import child_environ, span
from kta.store import controls as controls_store
from kta.store.base import Ops, Store, Task


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


def _limits(card: TaskCard, controls: Controls) -> Limits:
    """The card's wall budget (fixed when the card was written), else the controls' value
    for its type; the idle cut-off is the controls' no-response timeout."""
    wall = card.budget.get("wall_s") or controls.task_wall_s.get(card.type)
    if not wall:
        raise ConfigError(f"no wall budget for {card.type} in the card or the controls")
    return Limits(wall_s=int(wall), idle_s=controls.no_response_timeout_s)


def _env(ctx: Context, deps: Deps, token: str, task: Task, px: ProxyPlan) -> dict[str, str]:
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
    return guard.agent_env(values, ["{kta_bin}"], px.env)


def _spec(
    ctx: Context,
    deps: Deps,
    task: Task,
    card: TaskCard,
    profile: AgentProfile,
    run: str,
    bundle: workspace.Bundle,
    batch: str,
    arm: Arm,
    px: ProxyPlan,
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
        batch=batch,
        batch_files=sorted(bundle.batch),
        arm=arm.to_dict(),
        auth=auth,
        run_as=run_as,
        secret_env=[guard.TOKEN_VAR],
        notes_file="NOTES.md",
        nvidia_smi=host.nvidia_smi,
        watch=list(deps.config.watch),
        expect_version=profile.cli_version,
        proxy=px.spec(deps.config.secrets_file),
    )


def _renewing(ops: Ops, task: Task, stop: threading.Event, lease_s: int) -> None:
    owner = task.lease_owner or "agent-runner"
    while not stop.wait(lease_s / 6):  # renewed well inside the lease (budget.lease_s)
        ops.renew(task.id, owner, lease_s)


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
    deps: Deps,
    launch: Launch,
    events: list[AgentEvent],
    profile: AgentProfile,
    card: TaskCard,
    arm: Arm,
    controls: Controls,
) -> list[dict[str, Any]]:
    allowed = arms.commands(arm, profile.allowed_commands.get(card.type, []))
    report = launch.report
    out = compliance.from_events(events, str(report.get("workspace", "")), allowed)
    out += compliance.from_report(report, bool(controls.notes_required_sections))
    if launch.manifest and launch.workspace:
        mf = json.loads(deps.runner.read_blob(launch.manifest))
        after = _tar_hashes(deps.runner.read_blob(launch.workspace))
        out += compliance.context_changes(mf["workspace"]["files"], after)
    return out


def _prompt(
    batch: BatchRoot,
    mode: str,
    card: TaskCard,
    task: Task,
    retry: cards.RetryNote | None,
    note: str | None,
) -> str:
    def tpl(name: str) -> str:
        return batch.text(f"msg-{name}.md")

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


def _plan(
    ctx: Context,
    deps: Deps,
    task: Task,
    card: TaskCard,
    profile: AgentProfile,
    run: str,
    retry: cards.RetryNote | None,
    batch: BatchRoot,
    arm: Arm,
    controls: Controls,
    px: ProxyPlan,
) -> tuple[StepPlan, Limits]:
    """Bundle, launch spec and job inputs of one step."""
    bundle = workspace.build(
        deps.store,
        profile,
        batch,
        task_id=task.id,
        card_id=task.card_id,
        card_body=card,
        used={},
        retry=retry,
        policy=policy.context_policy(controls),
        arm=arm,
    )
    limits = _limits(card, controls)
    spec = _spec(ctx, deps, task, card, profile, run, bundle, batch.root.name, arm, px)
    inputs = {
        **_put_files(deps.runner, BUNDLE_WORKSPACE, bundle.workspace),
        **_put_files(deps.runner, BUNDLE_BATCH, bundle.batch),
    }
    cap = int(controls.hard_cap_factor * limits.wall_s)  # budget.hard_cap_factor
    return StepPlan(spec, inputs, {"pool": deps.config.pool}, hard_cap_s=cap), limits


def open_batch(
    ctx: Context, deps: Deps, card: TaskCard, state: sessions.SessionState | None, name: str | None
) -> BatchRoot:
    """The verified batch root of this step; a work unit never changes batch.

    Without an explicit batch, a configured batch built from another profile gives way to
    the card's own profile batch ``<name>-v<version>`` when it exists (D595)."""
    explicit = name is not None
    name = name or deps.config.batch
    if not name:
        raise ConfigError("no batch: pass batch= or set [agents] batch (kta agents env build)")
    if state is not None and state.batch and state.batch != name:
        raise ConfigError(f"work unit {card.work_unit} belongs to batch {state.batch}, not {name}")
    root = BatchRoot.open(deps.store, ctx.data_dir, name)
    if root.lock["profile"] != card.profile and not explicit and state is None:
        own = batches.profile_batch(AgentProfile.model_validate(deps.store.get(card.profile).body))
        if (batches.batches_root(ctx.data_dir) / own).exists():
            name, root = own, BatchRoot.open(deps.store, ctx.data_dir, own)
    if root.lock["profile"] != card.profile:
        raise ConfigError(
            f"batch {name} is built from profile {root.lock['profile']}, not the card's"
        )
    return root


def run_task(ctx: Context, task: Task) -> AgentRun:
    """The spec interface: run one step of a leased task and return its AgentRun."""
    return run_step(ctx, task).run


def run_step(
    ctx: Context,
    task: Task,
    *,
    retry: cards.RetryNote | None = None,
    note: str | None = None,
    deps: Deps | None = None,
    lane: str = "agent:1",
    batch: str | None = None,
) -> StepResult:
    """Run one step of ``task``, write its AgentRun, return it with its record ID (D302).

    ``retry`` carries the rejection of the previous attempt (from the orchestrator);
    ``note`` is a human's message, passed verbatim. ``batch`` (default: the task's
    ``info["batch"]``, then ``[agents] batch``) names the batch root; a batch that fails
    verification refuses the start (D322).
    """
    deps = deps or deps_from(ctx)
    card = TaskCard.model_validate(deps.store.get(task.card_id).body)
    profile = AgentProfile.model_validate(deps.store.get(card.profile).body)
    run = UlidFactory(ctx.clock).new()
    controls = controls_store.ensure(deps.store, deps.ops, ctx.clock.now())
    state, raw = sessions.load(deps.ops, card.work_unit)
    if state is not None and state.worker and state.worker != deps.config.worker:
        raise ConfigError(
            f"work unit {card.work_unit} lives on {state.worker}, not {deps.config.worker}"
        )
    info = arms.task_info(deps.ops, task.id)
    root = open_batch(ctx, deps, card, state, batch or info.get("batch"))
    arm = arms.arm_of(info)
    px = proxyplan.for_step(ctx.data_dir, deps.ops, profile.provider, deps.config.api_url)
    plan, limits = _plan(ctx, deps, task, card, profile, run, retry, root, arm, controls, px)
    adapter = adapter_for(profile, deps, plan)
    decision = sessions.decide(
        state,
        adapter.supports_resume,
        policy.loop_policy(controls),
        retry.signature if retry else None,
    )
    key = guard.ensure_key(Path(deps.config.token_key_file))
    token = guard.issue_token(key, task.id, ctx.clock.now() + timedelta(seconds=plan.hard_cap_s))
    prompt = _prompt(root, decision.mode, card, task, retry, note)
    stop = threading.Event()
    renewer = threading.Thread(
        target=_renewing, args=(deps.ops, task, stop, controls.lease_s), daemon=True
    )
    with span(
        "agent_session",
        f"{card.type} {card.work_unit}",
        lane,
        task_id=task.id,
        run=run,
        mode=decision.mode,
    ) as sp:
        env = _env(ctx, deps, token, task, px)
        renewer.start()
        try:
            events = _launch(adapter, decision, state, prompt, card, env, limits)
        finally:
            stop.set()
        launch = adapter.last_launch()
        assert launch is not None
        sp.set(session=launch.session, exit=launch.exit, worker=launch.worker)
    now = ctx.clock.now()
    proxyplan.outcome(deps.store, deps.ops, px, launch.exit, launch.report, now, task.id)
    return _record(
        ctx, deps, task, card, profile, launch, events, prompt, state, raw, decision, retry, arm,
        controls,
    )  # fmt: skip


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


def _agent_run(
    deps: Deps,
    card: TaskCard,
    profile: AgentProfile,
    launch: Launch,
    events: list[AgentEvent],
    arm: Arm,
    controls: Controls,
) -> AgentRun:
    usage = usage_of(events)
    return AgentRun(
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
        violations=violations_of(deps, launch, events, profile, card, arm, controls),
        arm=arm.to_dict(),
        proxy=launch.report.get("proxy"),
    )


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
    arm: Arm,
    controls: Controls,
) -> StepResult:
    usage = usage_of(events)
    body = _agent_run(deps, card, profile, launch, events, arm, controls)
    refs: dict[str, Any] = {
        "env": launch.env_manifest,
        "worker": launch.worker,
        "card": task.card_id,
        "profile": card.profile,
    }
    if arm.snapshot:
        refs["snapshot"] = arm.snapshot
    if controls.version_id:
        refs["controls"] = controls.version_id  # the knob values this step ran under (WP21)
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
        batch=str(launch.report.get("batch") or ""),
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


__all__ = ["Deps", "StepResult", "parse_session", "run_step", "run_task"]
