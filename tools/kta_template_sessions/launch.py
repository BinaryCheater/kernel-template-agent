"""Executor job program: run a persistent CLI in an external experiment package."""

from __future__ import annotations

import json
import os
import subprocess  # kta:allowed Executor job program, exports native DSH compressed frames
import sys
from pathlib import Path

from kta.agents import hostfs, provider_env
from kta.agents.adapters import codex_session
from kta.agents.launch import cli_identity, run_cli
from kta.agents.launchspec import (
    OUT_ARCHIVE,
    OUT_MANIFEST,
    OUT_REPORT,
    OUT_STDOUT,
    OUT_WORKSPACE,
    LaunchSpec,
)
from kta.core.errors import ConfigError


def step(job: Path, payload: dict) -> dict:
    spec = LaunchSpec.from_dict(payload["spec"])
    ws, state = Path(payload["workspace"]), Path(payload["state"])
    if not ws.is_absolute() or not state.is_absolute() or not (ws / "TASK.md").is_file():
        raise ConfigError("external workspace must contain TASK.md; paths must be absolute")
    if hostfs.upward_agents_md(ws):
        raise ConfigError("experiment package has an ancestor AGENTS.md")
    home = state / "home"
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    creds = []
    if spec.provider == "codex":
        provider_env.inherit_connection(home, provider_env.codex_home())
        creds = hostfs.link_credentials(home, provider_env.codex_home(), ["auth.json"])
        config = home / "config.toml"
        if not config.exists():
            config.write_text("")
    # Personal instructions/skills are never copied. Restrict file reads by instruction;
    # full-access is explicitly authorized for experimental remote tool use.
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        in (
            "PATH",
            "HOME",
            "USER",
            "TMPDIR",
            "LANG",
            "LC_ALL",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
        )
    }
    env.update(spec.env)
    if spec.provider == "codex":
        env["CODEX_HOME"] = str(home)
    else:
        env["DSH_PERMISSION_MODE"] = "danger-full-access"
    argv = list(spec.argv)
    identity = cli_identity(argv)
    out = job / "out"
    out.mkdir(exist_ok=True)
    manifest = {
        "workspace": str(ws),
        "home": str(home),
        "cli": identity,
        "argv": argv,
        "model": "gpt-6.1-sol" if spec.provider == "codex" else "deepseek-flash",
        "reasoning_effort": "high",
        "isolation": "danger-full-access; instruction-only workspace/source/network isolation",
        "wall_s": spec.wall_s,
        "idle_s": spec.idle_s,
        "mode": spec.mode,
        "session": spec.session,
        "service_tier": "not enabled",
    }
    hostfs.write_json(out / OUT_MANIFEST, manifest)
    outcome = run_cli(spec, argv, env, ws, home, job)
    sid = codex_session.session_from_stdout((out / OUT_STDOUT).read_text().splitlines())
    if spec.provider == "deepseek_harness":
        rows = []
        for line in (out / OUT_STDOUT).read_text().splitlines():
            if line.startswith("{"):
                rows.append(json.loads(line))
        sid = next((r["sessionId"] for r in rows if r.get("type") == "session"), None)
    sid = sid or spec.session
    resolved = []
    if spec.provider == "deepseek_harness":
        electron = Path(argv[0]).parents[4] / "MacOS/DeepSeek Harness"
        export = subprocess.run(
            [str(electron), str(job / "export-dsh.cjs"), str(home / "sessions")],
            capture_output=True,
            text=True,
            check=True,
            env={**env, "ELECTRON_RUN_AS_NODE": "1"},
            timeout=30,
        )
        resolved = [json.loads(line) for line in export.stdout.splitlines()]
    native = [
        str(p) for p in (home / "sessions").rglob("*") if p.is_file() and sid and sid in str(p)
    ]
    # Archive only native transcripts. Provider configuration/credentials remain local.
    archive, flagged = hostfs.pack(
        home, exclude_top=tuple(p.name for p in home.iterdir() if p.name != "sessions")
    )
    (out / OUT_ARCHIVE).write_bytes(archive)
    workspace, ws_flagged = hostfs.pack(ws, exclude_top=(".git",))
    (out / OUT_WORKSPACE).write_bytes(workspace)
    report = {
        "session": sid,
        "resolved_native": [r for r in resolved if r["session"] == sid],
        "exit": outcome.exit,
        "exit_code": outcome.exit_code,
        "wall_s": outcome.wall_s,
        "cli_version": identity["version"],
        "native_transcripts": native,
        "home": str(home),
        "workspace": str(ws),
        "secret_members": flagged + ws_flagged,
        "credentials_refreshed": hostfs.restore_credentials(home, provider_env.codex_home(), creds),
    }
    hostfs.write_json(out / OUT_REPORT, report)
    return report


if __name__ == "__main__":
    job = Path(sys.argv[1]).resolve()
    step(job, json.loads((job / "launch.json").read_text()))
