"""Sealed campaign sessions and a live queue-adjusted active-time watchdog."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess  # kta:allowed campaign Executor job program only
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from kta.core.errors import ConfigError


def queue_account(rows: list[dict[str, Any]], slot: str, start_s: float, now_s: float) -> dict:
    """Validate the sole blocking request and count each queue interval exactly once."""
    request = None
    queued = None
    verified = None
    queue_s = 0.0
    submissions = 0
    seen: set[str] = set()
    submissions_seen: set[str] = set()
    phase = None
    last_s = float("-inf")
    for row in rows:
        at = float(row["at_s"])
        observed = float(row.get("observed_at_s", at))
        if row["slot"] != slot or observed < last_s or at > observed + 2 or observed > now_s + 2:
            raise ConfigError("queue ledger identity/order/clock mismatch")
        last_s = observed
        event, rid = row["event"], row["request_id"]
        if event == "blocked":
            if request is not None or rid in seen:
                raise ConfigError("overlapping or repeated evaluator request")
            request = rid
            seen.add(rid)
            submissions_seen.add(row.get("submission_id", rid))
            submissions = len(submissions_seen)
            phase = "blocked"
        elif event in ("queued", "waiting", "acquired", "completed"):
            if request != rid or not row.get("worker") or not row.get("job_id"):
                raise ConfigError("queue event lacks sole blocking request/worker/job evidence")
            if event == "queued":
                if phase != "blocked":
                    raise ConfigError("repeated queue start")
                queued, verified = at, at
                phase = "queued"
            elif event == "waiting":
                if phase != "queued":
                    raise ConfigError("queue heartbeat without open wait")
                verified = at
            elif event == "acquired":
                if phase != "queued" or queued is None:
                    raise ConfigError("lock acquired without queue start")
                queue_s += max(0.0, at - max(start_s, queued))
                queued = None
                phase = "acquired"
            else:
                if queued is not None:
                    queue_s += max(0.0, (verified or queued) - max(start_s, queued))
                request, queued, phase = None, None, None
        else:
            raise ConfigError(f"unknown evaluator queue event: {event}")
    if queued is not None:
        queue_s += max(0.0, (verified or queued) - max(start_s, queued))
    return {"queue_s": queue_s, "submissions": submissions, "blocked_request_id": request}


def ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw = path.read_text()
    # The evaluator appends newline-terminated records; a concurrent partial write is pending.
    lines = raw.splitlines() if raw.endswith("\n") else raw.splitlines()[:-1]
    return [json.loads(line) for line in lines]


def snapshot(config: dict, now_s: float) -> dict:
    rows = ledger(Path(config["ledger"]))
    accounting = queue_account(rows, config["slot"], config["start_s"], now_s)
    wall_s = max(0.0, now_s - config["start_s"])
    active_s = config.get("prior_active_s", 0.0) + wall_s - accounting["queue_s"]
    if accounting["queue_s"] > wall_s + 0.001:
        raise ConfigError("verified queue exceeds session wall time")
    return {
        **accounting,
        "slot": config["slot"],
        "at_s": now_s,
        "wall_s": wall_s,
        "active_s": active_s,
        "remaining_active_s": max(0.0, config["budget_s"] - active_s),
    }


def atomic_json(path: Path, value: dict) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2))
    pending.replace(path)


def campaign_run_cli(spec, argv, env, ws, home, job):
    """Replacement only inside the campaign Executor job; no fixed wall/idle timeout."""
    from kta.agents.adapters.codex_session import session_files, session_from_stdout
    from kta.agents.launch import Outcome

    config = json.loads((job / "campaign-watchdog.json").read_text())
    config["start_s"] = time.time()
    out = job / "out"
    checkpoint = Path(config["checkpoint"])
    status_path = Path(config["remaining"])
    outcome = Outcome()
    interrupted = False

    def stop_requested(signum, frame):
        nonlocal interrupted
        interrupted = True

    old_int = signal.signal(signal.SIGINT, stop_requested)
    old_term = signal.signal(signal.SIGTERM, stop_requested)
    proc = None
    started = time.monotonic()
    with (
        (job / "prompt.md").open("rb") as prompt,
        (out / "stdout.jsonl").open("wb") as stdout,
        (out / "stderr.txt").open("wb") as stderr,
    ):
        try:
            proc = subprocess.Popen(
                argv,
                cwd=ws,
                env=env,
                stdin=prompt,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            while True:
                now = time.time()
                current = snapshot(config, now)
                atomic_json(status_path, current)
                raw = (out / "stdout.jsonl").read_text()
                sid = session_from_stdout(raw.splitlines()) or spec.session
                if sid:
                    atomic_json(
                        checkpoint,
                        {
                            **current,
                            "session": sid,
                            "home": str(home),
                            "native_transcripts": [
                                str(p) for p in session_files(home) if sid in str(p)
                            ],
                        },
                    )
                reason = "cancelled" if interrupted else None
                if current["remaining_active_s"] <= 0:
                    reason = "active_budget"
                elif (
                    current["submissions"] >= config["max_submissions"]
                    and not current["blocked_request_id"]
                ):
                    reason = "submission_budget"
                last_output = max((out / n).stat().st_mtime for n in ("stdout.jsonl", "stderr.txt"))
                idle_queue = queue_account(
                    ledger(Path(config["ledger"])), config["slot"], last_output, now
                )["queue_s"]
                if now - last_output - idle_queue > config["idle_s"]:
                    reason = "idle"
                if proc.poll() is not None:
                    outcome.exit_code = proc.returncode
                    outcome.exit = "ok" if proc.returncode == 0 else "nonzero"
                    break
                if reason:
                    outcome.exit = reason
                    os.killpg(proc.pid, signal.SIGINT)
                    try:
                        proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGTERM)
                        proc.wait(timeout=10)
                    outcome.exit_code = proc.returncode
                    break
                time.sleep(0.25)
        except BaseException:
            # Protocol errors and caller interruption must not leave a paid native process alive.
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                proc.wait(timeout=20)
            raise
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)
            outcome.wall_s = time.monotonic() - started
            atomic_json(status_path, snapshot(config, time.time()))
    return outcome


def worker(job: Path) -> None:
    module_spec = importlib.util.spec_from_file_location(
        "campaign_native_launch", job / "experiment-launch.py"
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.__dict__["run_cli"] = campaign_run_cli
    module.step(job, json.loads((job / "launch.json").read_text()))
    manifest = json.loads((job / "out/manifest.json").read_text())
    manifest["campaign_watchdog"] = json.loads((job / "campaign-watchdog.json").read_text())
    manifest["campaign_watchdog"]["idle_policy"] = "same active cap; verified queue excluded"
    atomic_json(job / "out/manifest.json", manifest)


def make_adapter(runner, plan, workspace: Path, state: Path, budget: dict):
    from kta.agents.adapters.base import collect
    from kta.agents.base import Launch
    from kta.core.jobs import Job
    from kta_template_sessions.adapter import RecordedCodexAdapter

    class CampaignCodexAdapter(RecordedCodexAdapter):
        def _run(self, spec, prompt, limits, env) -> Launch:
            spec = replace(
                spec, env={**spec.env, **env}, wall_s=limits.wall_s, idle_s=limits.idle_s
            )
            payload = {
                "spec": json.loads(spec.to_json()),
                "workspace": str(self.workspace),
                "state": str(self.state),
            }
            source = Path(__file__).resolve().parents[1] / "kta_template_sessions/launch.py"
            inputs = {
                "launch.json": json.dumps(payload).encode(),
                "prompt.md": prompt.encode(),
                "campaign-watchdog.json": json.dumps(budget).encode(),
                "session-launch.py": Path(__file__).read_bytes(),
                "experiment-launch.py": source.read_bytes(),
            }
            result = self.runner.run(
                Job(
                    kind="shell",
                    argv=["{python}", "{job}/session-launch.py", "--job", "{job}"],
                    requires={"pool": "agent"},
                    timeout_s=604800,
                    inputs={
                        k: self.runner.put_blob(v, "application/octet-stream")
                        for k, v in inputs.items()
                    },
                )
            )
            for name, blob in result.outputs.items():
                if name in ("stdout.jsonl", "stderr.txt", "report.json", "manifest.json"):
                    (self.state / spec.run / name).write_bytes(self.runner.read_blob(blob))
            self._last = collect(self.runner, result, spec.session)
            return self._last

    return CampaignCodexAdapter(
        runner,
        plan,
        [str(Path.home() / ".local/bin/codex")],
        [
            "--json",
            "--skip-git-repo-check",
            "-c",
            'sandbox_mode="danger-full-access"',
            "-c",
            'approval_policy="never"',
            "-c",
            'web_search="disabled"',
            "-c",
            "features.multi_agent=false",
            "-c",
            "features.multi_agent_v2=false",
        ],
        "gpt-6.1-sol",
        "high",
        workspace=workspace,
        state=state,
    )


def run(root: Path, slot: str, resume: bool = False) -> dict:
    """Only this campaign path removes the ordinary launcher's fixed wall-clock stop."""
    import fcntl

    from kta.agents.adapters.base import StepPlan
    from kta.agents.adapters.codex_session import usage_of
    from kta.agents.base import Limits
    from kta.agents.launchspec import LaunchSpec
    from kta.cli.exec_cmds import command
    from kta.cli.main import context_from_environ
    from kta.core.ids import UlidFactory
    from kta.core.schemas import Actor, Record
    from kta.core.schemas.process import AgentProfile, AgentRun
    from kta.core.trace import span
    from kta.orchestrator.pipeline import Services
    from kta_source_replay.seal import read, verify_seal

    if slot not in {f"run-{i:02d}" for i in range(1, 10)}:
        raise ConfigError("unknown campaign slot")
    with (root / "slots" / slot / "session.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConfigError("this implementation session is already active") from exc
        if (root / "HALT.json").exists():
            raise ConfigError("campaign halted; preserve current state and diagnose")
        sealed = verify_seal(root, slot, pristine=not resume)
        settings = read(Path(sealed["inventory"]["settings"]["path"]))
        slots = [row["slot"] for row in settings["schedule"]]
        if slot not in slots:
            raise ConfigError("slot is not in the sealed schedule")
        if settings["implementation_concurrency"] == 1:
            for prior in slots[: slots.index(slot)]:
                if not (root / "slots" / prior / "model-closed.json").is_file():
                    raise ConfigError("earlier scheduled slot is not closed")
        home_root = root / "slots" / slot
        state, workspace = home_root / "state", home_root / "workspace"
        checkpoint = home_root / "native-session.json"
        remaining = home_root / "remaining-active.json"
        progress = read(remaining) if remaining.exists() else {}
        prior_native = read(checkpoint) if checkpoint.exists() else None
        if (home_root / "model-closed.json").exists():
            raise ConfigError("slot model session is already closed")
        if bool(prior_native) != resume:
            raise ConfigError("fresh start/resume does not match retained native session")
        cache = read(Path(sealed["inventory"]["cache_settings"]["path"]))["slots"][slot]
        for value in cache["local_env"].values():
            path = Path(value)
            if (
                not path.is_absolute()
                or home_root not in path.parents
                or any(p.is_symlink() for p in (path, *path.parents))
            ):
                raise ConfigError("compiler cache escapes its slot")
            if not resume and path.exists() and any(path.iterdir()):
                raise ConfigError("fresh slot compiler cache is nonempty")
            path.mkdir(parents=True, exist_ok=True)
        state.mkdir(exist_ok=True)
        if not resume and (state / "home").exists():
            raise ConfigError("fresh slot native home already exists")
        budget = {
            "slot": slot,
            "ledger": str(home_root / "queue-events.jsonl"),
            "checkpoint": str(checkpoint),
            "remaining": str(remaining),
            "budget_s": settings["budget"]["active_seconds"],
            "max_submissions": settings["budget"]["target_evaluation_submissions"],
            "idle_s": settings["budget"]["active_seconds"],
            "prior_active_s": progress.get("active_s", 0.0),
            "start_s": time.time(),
        }
        current = snapshot(budget, time.time())
        if current["remaining_active_s"] <= 0 or current["blocked_request_id"]:
            raise ConfigError("no active budget remains or a prior evaluator request is still open")
        prompt = (workspace / "PROMPT.md").read_text()
        with command(
            context_from_environ, "source-replay-session", f"controlled replay {slot}"
        ) as ctx:
            if ctx.env != "dev":
                raise ConfigError("campaign runs require KTA_ENV=dev")
            svc = Services.open(ctx)
            run_id = UlidFactory(ctx.clock).new()
            turn = state / run_id
            turn.mkdir()
            spec = LaunchSpec(
                provider="codex", work_unit=slot, run=run_id, mode="start", gpu_guard=False
            )
            adapter = make_adapter(svc.executor, StepPlan(spec), workspace, state, budget)
            before = usage_of(adapter.parse_session(state / "home")) if resume else None
            env = {
                **cache["local_env"],
                "KTA_SOURCE_REPLAY_ROOT": str(root),
                "KTA_SOURCE_REPLAY_SLOT": slot,
                "SOURCE_REPLAY_CLI": str(
                    Path(__file__).resolve().parents[2] / "scripts/source-replay"
                ),
            }
            (turn / "prompt.md").write_text(prompt)
            (turn / "settings.json").write_text(
                json.dumps(
                    {
                        "model": "gpt-6.1-sol",
                        "effort": "high",
                        "fast_service": False,
                        "env": env,
                        "budget": budget,
                        "mode": "resume" if resume else "start",
                        "seal": sealed,
                    },
                    indent=2,
                )
            )
            atomic_json(
                home_root / "session-started.json",
                {"slot": slot, "at_s": time.time(), "prior_active_s": budget["prior_active_s"]},
            )
            with span("agent_session", f"controlled source replay {slot}", "agent:source-replay"):
                if resume:
                    assert prior_native is not None
                    events = list(
                        adapter.resume(
                            prior_native["session"], prompt, workspace, env, Limits(1800, 1800)
                        )
                    )
                else:
                    _, iterator = adapter.start(prompt, workspace, env, Limits(1800, 1800))
                    events = list(iterator)
            launch = adapter.last_launch()
            assert launch is not None
            usage = usage_of(events)
            accounting = read(remaining)
            totals = {k: getattr(usage, k) for k in ("tokens_in", "tokens_out", "cached_in")}
            deltas = {
                k: (v - getattr(before, k))
                if v is not None and before is not None and getattr(before, k) is not None
                else v
                if before is None
                else None
                for k, v in totals.items()
            }
            actor = Actor(type="tool", name="source replay controlled session")
            profile = AgentProfile(
                name="source-replay",
                version=1,
                provider="codex",
                cli_version=str(launch.report.get("cli_version")),
                auth="subscription",
                isolation="config",
                model="gpt-6.1-sol",
                reasoning_effort="high",
                prefix_template="",
            )
            profile_id = svc.store.put(Record.new("AgentProfile", profile, actor, ctx.clock.now()))
            body = AgentRun(
                session=launch.session,
                profile=profile_id,
                cli_version=str(launch.report.get("cli_version")),
                model="gpt-6.1-sol",
                tokens_in=deltas["tokens_in"],
                tokens_out=deltas["tokens_out"],
                tokens_cached=deltas["cached_in"],
                tool_calls=sum(e.kind == "tool_call" for e in events),
                session_mode="resume" if resume else "start",
                session_archive=launch.archive,
                wall_s=launch.wall_s,
                tool_s=0,
                queue_s=accounting["queue_s"],
                exit=launch.exit,
                context_manifest=launch.manifest,
                arm={
                    "slot": slot,
                    "tool_time_status": "native transcript retained; not reconstructed",
                    "usage_scope": "per-turn differences of native cumulative totals",
                },
            )
            rid = svc.store.put(
                Record.new(
                    "AgentRun",
                    body,
                    Actor(type="tool", name="source replay controlled session"),
                    ctx.clock.now(),
                    refs={"worker": launch.worker, "env": launch.env_manifest},
                    blobs={"stdout": launch.stdout, "workspace": launch.workspace},
                )
            )
            info = {
                "session": launch.session,
                "agent_run": rid,
                "exit": launch.exit,
                "accounting": accounting,
                "per_turn_usage": deltas,
                "tokens_in": usage.tokens_in,
                "tokens_out": usage.tokens_out,
                "cached_input": usage.cached_in,
                "cost": None,
                "cost_reason": "native CLI does not report billing",
                "native_transcripts": launch.report["native_transcripts"],
                "previous_usage": before.__dict__ if before else None,
            }
            (turn / "result.json").write_text(json.dumps(info, indent=2))
            for name, blob in (
                ("native-session.tar.gz", launch.archive),
                ("stdout.jsonl", launch.stdout),
            ):
                (turn / name).write_bytes(svc.executor.read_blob(blob))
            if launch.exit != "cancelled":
                atomic_json(home_root / "model-closed.json", info)
            return info


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--job":
        worker(Path(sys.argv[2]))
    else:
        raise SystemExit("Use scripts/source-replay session DATA_ROOT --slot RUN")
