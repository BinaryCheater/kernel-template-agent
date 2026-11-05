"""Launch or resume recorded Template-replay implementation sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kta.agents.adapters.base import StepPlan
from kta.agents.adapters.codex_session import usage_of
from kta.agents.base import Limits
from kta.agents.launchspec import LaunchSpec
from kta.agents.runner import tool_seconds
from kta.cli.exec_cmds import command
from kta.cli.main import context_from_environ
from kta.core.errors import ConfigError
from kta.core.ids import UlidFactory
from kta.core.schemas import Actor, Record
from kta.core.schemas.process import AgentProfile, AgentRun
from kta.core.trace import span
from kta.orchestrator.pipeline import Services

from .adapter import RecordedCodexAdapter
from .dsh import RecordedDshAdapter, patch


def execute(args: argparse.Namespace) -> dict:
    ws, state = args.workspace.resolve(), args.state.resolve()
    if not (ws / "TASK.md").is_file():
        raise ConfigError("workspace needs TASK.md")
    state.mkdir(parents=True, exist_ok=True)
    previous_path = state / "session.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
    if args.mode == "start" and previous is not None:
        raise ConfigError("session already exists; resume it or use a fresh state directory")
    if args.mode == "resume" and (not previous or not previous.get("session")):
        raise ConfigError("resume requires session.json with a native session identity")
    if previous and previous.get("provider", "codex") != args.provider:
        raise ConfigError("resume provider differs from saved native session")
    if previous and previous.get("workspace", str(ws)) != str(ws):
        raise ConfigError("resume workspace differs from saved native session")
    prompt = args.prompt.read_text()
    with command(context_from_environ, "template-session", f"Template replay {args.mode}") as ctx:
        if ctx.env != "dev":
            raise ConfigError("experiment session launcher requires KTA_ENV=dev")
        svc = Services.open(ctx)
        run = UlidFactory(ctx.clock).new()
        spec = LaunchSpec(
            provider=args.provider, work_unit=state.name, run=run, mode="start", gpu_guard=False
        )
        flags = [
            "--json",
            "--skip-git-repo-check",
            "-c",
            'sandbox_mode="danger-full-access"',
            "-c",
            'approval_policy="never"',
            "-c",
            'web_search="disabled"',
        ]
        if args.provider == "deepseek_harness":
            flags = []
            (state / "dsh.patch.json").write_text(json.dumps(patch(state), indent=2))
        adapter_class = RecordedCodexAdapter if args.provider == "codex" else RecordedDshAdapter
        adapter = adapter_class(
            svc.executor,
            StepPlan(spec=spec, requires={"pool": "agent"}),
            [str(args.codex.expanduser() if args.provider == "codex" else args.dsh)],
            flags,
            "gpt-6.1-sol" if args.provider == "codex" else "deepseek-flash",
            "high",
            workspace=ws,
            state=state,
        )
        profile = AgentProfile(
            name=f"template-replay-{args.provider}",
            version=1,
            provider=args.provider,
            cli_version="runtime",
            auth="subscription",
            isolation="config",
            model=adapter.model,
            reasoning_effort=adapter.effort,
            cli_flags=flags,
            prefix_template="",
        )
        actor = Actor(type="tool", name="template-replay recorded sessions")
        profile_id = svc.store.put(Record.new("AgentProfile", profile, actor, ctx.clock.now()))
        turn = state / run
        turn.mkdir()
        (turn / "prompt.md").write_text(prompt)
        (turn / "settings.json").write_text(
            json.dumps(
                {
                    "model": adapter.model,
                    "effort": adapter.effort,
                    "flags": flags,
                    "argv": adapter.start_argv()
                    if args.mode == "start"
                    else adapter.resume_argv(previous["session"] if previous else ""),
                    "workspace": str(ws),
                    "wall_s": args.wall_s,
                    "idle_s": args.idle_s,
                    "mode": args.mode,
                    "previous_session": previous.get("session") if previous else None,
                    "status": "launching",
                    "isolation": "instruction-only workspace and source/network isolation",
                },
                indent=2,
            )
        )
        started_at = ctx.clock.now()
        with span("agent_session", "Template replay implementation", "agent:template-replay") as sp:
            limits = Limits(wall_s=args.wall_s, idle_s=args.idle_s)
            if args.mode == "resume":
                assert previous is not None
                events = list(adapter.resume(previous["session"], prompt, ws, {}, limits))
            else:
                _, iterator = adapter.start(prompt, ws, {}, limits)
                events = list(iterator)
            launch = adapter.last_launch()
            assert launch is not None
            sp.set(session=launch.session, exit=launch.exit)
        ended_at = ctx.clock.now()
        usage = usage_of(events)
        totals = {
            "input": usage.tokens_in,
            "output": usage.tokens_out,
            "cached_input": usage.cached_in,
        }
        prior = previous.get("usage", {}) if previous and args.provider == "codex" else {}
        delta = {
            k: v - prior.get(k, 0) if v is not None and prior.get(k, 0) is not None else None
            for k, v in totals.items()
        }
        body = AgentRun(
            session=launch.session,
            profile=profile_id,
            cli_version=launch.report["cli_version"],
            model=adapter.model,
            tokens_in=delta["input"],
            tokens_out=delta["output"],
            tokens_cached=delta["cached_input"],
            tool_calls=sum(e.kind == "tool_call" for e in events),
            session_mode=args.mode,
            session_archive=launch.archive,
            wall_s=launch.wall_s,
            tool_s=tool_seconds(events),
            queue_s=launch.queue_s,
            exit=launch.exit,
            context_manifest=launch.manifest,
            arm={
                "name": "template-replay",
                "accounting": "per-turn differences of native cumulative totals"
                if args.provider == "codex"
                else "sum of per-step native usage for this turn",
                "tool_time_status": "parsed from native transcript"
                if args.provider == "codex"
                else "unavailable; required tool_s field is zero",
            },
        )
        rid = svc.store.put(
            Record.new(
                "AgentRun",
                body,
                actor,
                ctx.clock.now(),
                refs={"worker": launch.worker, "env": launch.env_manifest, "profile": profile_id},
                blobs={
                    "stdout": launch.stdout,
                    "workspace": launch.workspace,
                    "prompt": svc.executor.put_blob(prompt.encode(), "text/markdown"),
                },
            )
        )
        for name, blob in (
            ("native-session.tar.gz", launch.archive),
            ("stdout.jsonl", launch.stdout),
            ("manifest.json", launch.manifest),
        ):
            (turn / name).write_bytes(svc.executor.read_blob(blob))
        info = {
            "session": launch.session,
            "provider": args.provider,
            "workspace": str(ws),
            "agent_run": rid,
            "turn": str(turn),
            "exit": launch.exit,
            "native_transcripts": launch.report["native_transcripts"],
            "model": adapter.model,
            "effort": adapter.effort,
            "usage": {
                "input": usage.tokens_in,
                "output": usage.tokens_out,
                "cached_input": usage.cached_in,
                "scope": "cumulative session totals"
                if args.provider == "codex"
                else "per-turn sum of native step usage",
                "cost": None,
                "cost_status": "not reported by native harness",
            },
            "per_turn_usage": delta,
            "native_usage_details": [e.data for e in events if e.kind == "usage"][-1:],
            "resolved_native": launch.report.get("resolved_native", []),
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "wall_s": launch.wall_s,
            "queue_s": launch.queue_s,
        }
        (turn / "result.json").write_text(json.dumps(info, indent=2))
        if launch.session and launch.report["native_transcripts"]:
            previous_path.write_text(json.dumps(info, indent=2))
        return info


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("start", "resume"))
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--prompt", type=Path, required=True)
    p.add_argument("--provider", choices=("codex", "deepseek_harness"), default="codex")
    p.add_argument(
        "--dsh",
        type=Path,
        default=Path("/Applications/DeepSeek Harness.app/Contents/Resources/runtime/cli/bin/dsh"),
    )
    p.add_argument("--codex", type=Path, default=Path("~/.local/bin/codex"))
    p.add_argument("--wall-s", type=int, default=3600)
    p.add_argument("--idle-s", type=int, default=600)
    result = execute(p.parse_args())
    print(json.dumps(result, indent=2))
    return 0 if result["exit"] == "ok" and result["session"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
