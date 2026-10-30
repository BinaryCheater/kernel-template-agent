"""``python -m kta.agents.launch <job dir>``: the agent step, run as an Executor job.

This is the job program on the agent host (decision D300). It prepares the work unit's
workspace and isolated CLI home, writes the context manifest, starts the agent CLI with
a whitelisted environment, enforces the wall and idle limits and the GPU guard, commits
the workspace, and packs the session files without credentials. Everything it returns
goes under the job's ``out/`` and reaches the control plane by hash.

Like the Executor's worker, this program starts processes itself; that is why it is the
one agents file that imports subprocess (gate "subprocess outside Executor").
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess  # kta:allowed job program on the agent host, started by the Executor (D300)
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kta.agents import hostfs
from kta.agents.adapters import codex_session
from kta.agents.launchspec import (
    HOME,
    KTA_BIN,
    OUT_ARCHIVE,
    OUT_MANIFEST,
    OUT_REPORT,
    OUT_STDERR,
    OUT_STDOUT,
    OUT_WORKSPACE,
    PROMPT_FILE,
    SPEC_FILE,
    WORKSPACE,
    LaunchSpec,
)
from kta.core.errors import ConfigError, KtaError

POLL_S = 1.0
GPU_POLL_S = 5.0
KILL_GRACE_S = 10.0
GIT_ENV = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "kta-runner",
    "GIT_AUTHOR_EMAIL": "kta-runner@localhost",
    "GIT_COMMITTER_NAME": "kta-runner",
    "GIT_COMMITTER_EMAIL": "kta-runner@localhost",
}


@dataclass
class Outcome:
    exit: str = "ok"
    exit_code: int | None = None
    wall_s: float = 0.0
    gpu: list[dict[str, Any]] = field(default_factory=lambda: list[dict[str, Any]]())


def data_root(job_dir: Path) -> Path:
    """The agent host's data dir: jobs run in ``<data>/worker/jobs/<id>`` (worker.py)."""
    if job_dir.parent.name != "jobs" or job_dir.parents[1].name != "worker":
        raise ConfigError(f"launcher must run in a worker job directory, got {job_dir}")
    return job_dir.parents[2]


def _sub(value: str, home: Path, workspace: Path) -> str:
    kta_bin = str(Path(sys.executable).parent)
    return (
        value.replace(HOME, str(home)).replace(WORKSPACE, str(workspace)).replace(KTA_BIN, kta_bin)
    )


def _git(workspace: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(workspace), *args],
        capture_output=True,
        text=True,
        env={**GIT_ENV, "PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        timeout=120,
    )
    if proc.returncode != 0:
        raise ConfigError(f"git {' '.join(args)} failed: {proc.stderr.strip()[-500:]}")
    return proc.stdout.strip()


def commit_workspace(workspace: Path, message: str) -> str:
    if not (workspace / ".git").exists():
        _git(workspace, "init", "-q", "-b", "main")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-q", "--allow-empty", "-m", message)
    return _git(workspace, "rev-parse", "HEAD")


def cli_identity(argv: list[str]) -> dict[str, Any]:
    """Path, hash and ``--version`` of the CLI (argv up to its subcommand)."""
    cut = argv.index("exec") if "exec" in argv else 1
    prefix = argv[:cut]
    exe = Path(prefix[-1]) if prefix else Path(argv[0])
    try:
        proc = subprocess.run([*prefix, "--version"], capture_output=True, text=True, timeout=30)
        version = (proc.stdout or proc.stderr).strip()
    except OSError as exc:
        version = f"unavailable: {exc}"
    return {
        "command": prefix,
        "sha256": hostfs.sha256_file(exe) if exe.is_file() else None,
        "version": version,
    }


def _ancestors(pid: int) -> list[int]:
    out: list[int] = []
    for _ in range(64):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            break
        pid = int(stat.rsplit(")", 1)[1].split()[1])
        if pid <= 1:
            break
        out.append(pid)
    return out


def _owner(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("Uid:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def gpu_offenders(nvidia_smi: str, root_pid: int, uid: int | None) -> list[dict[str, Any]]:
    """GPU compute processes started under the agent (or owned by its user): forbidden."""
    try:
        proc = subprocess.run(
            [nvidia_smi, "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    found: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if root_pid in _ancestors(pid) or (uid is not None and _owner(pid) == uid):
            found.append({"pid": pid, "process": parts[1] if len(parts) > 1 else ""})
    return found


def _kill_tree(proc: subprocess.Popen[bytes], run_as: str | None) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            if run_as:
                # The CLI runs as another user; signal its group as that user (sudoers
                # rule written by scripts/kta-agent-setup).
                subprocess.run(
                    ["sudo", "-n", "-u", run_as, "kill", f"-{int(sig)}", f"-{proc.pid}"], timeout=30
                )
            else:
                os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
            pass  # kta:allowed the group already ended
        try:
            proc.wait(timeout=KILL_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            continue


def _latest_mtime(paths: list[Path], home: Path) -> float:
    candidates = [*paths, *codex_session.session_files(home)[-3:]]
    return max((p.stat().st_mtime for p in candidates if p.exists()), default=0.0)


def _uid_of(user: str | None) -> int | None:
    if user is None:
        return os.getuid()
    import pwd

    try:
        return pwd.getpwnam(user).pw_uid
    except KeyError:
        return None


def run_cli(
    spec: LaunchSpec, argv: list[str], env: dict[str, str], ws: Path, home: Path, job: Path
) -> Outcome:
    out_dir = job / "out"
    cmd = argv
    if spec.run_as:
        # umask 007: files the agent creates stay group-writable for the launcher, which
        # commits, archives and finally removes them (scripts/kta-agent-setup).
        pairs = [f"{k}={v}" for k, v in env.items()]
        umask = ["/bin/sh", "-c", 'umask 007; exec "$@"', "kta-agent"]
        cmd = ["sudo", "-n", "-u", spec.run_as, "--", *umask, "env", "-i", *pairs, *argv]
    stdout_path, stderr_path = out_dir / OUT_STDOUT, out_dir / OUT_STDERR
    outcome = Outcome()
    start = time.monotonic()
    uid = _uid_of(spec.run_as) if spec.run_as else None
    with (
        (job / PROMPT_FILE).open("rb") as stdin,
        stdout_path.open("wb") as so,
        stderr_path.open("wb") as se,
    ):
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=ws,
                env=None if spec.run_as else env,
                stdin=stdin,
                stdout=so,
                stderr=se,
                start_new_session=True,
            )
        except OSError as exc:
            return Outcome(exit="start_failed", gpu=[{"error": str(exc)}])
        next_gpu = start + GPU_POLL_S
        while True:
            try:
                outcome.exit_code = proc.wait(timeout=POLL_S)
                outcome.exit = "ok" if outcome.exit_code == 0 else "nonzero"
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            if now - start > spec.wall_s:
                outcome.exit = "wall"
            elif (
                now - start > spec.idle_s
                and time.time() - _latest_mtime([stdout_path, stderr_path], home) > spec.idle_s
            ):
                outcome.exit = "idle"
            elif spec.gpu_guard and now >= next_gpu:
                next_gpu = now + GPU_POLL_S
                outcome.gpu = gpu_offenders(spec.nvidia_smi, proc.pid, uid)
                if outcome.gpu:
                    outcome.exit = "gpu_guard"
            if outcome.exit in ("wall", "idle", "gpu_guard"):
                _kill_tree(proc, spec.run_as)
                outcome.exit_code = proc.returncode
                break
    outcome.wall_s = time.monotonic() - start
    return outcome


def _session(spec: LaunchSpec, stdout: Path, home: Path, before: set[Path]) -> str | None:
    if spec.mode == "resume" and spec.session:
        return spec.session
    sid = codex_session.session_from_stdout(stdout.read_text("utf-8", "replace").splitlines())
    if sid:
        return sid
    new = [p for p in codex_session.session_files(home) if p not in before]
    return codex_session.session_id_of(new[-1]) if new else None


def step(job: Path, spec: LaunchSpec) -> dict[str, Any]:
    data = data_root(job)
    ws, home = data / "work" / spec.work_unit, data / "agent-home" / spec.work_unit
    out = job / "out"
    report: dict[str, Any] = {
        "run": spec.run,
        "mode": spec.mode,
        "workspace": str(ws),
        "home": str(home),
    }
    upward = hostfs.upward_agents_md(ws)
    if upward:
        return {**report, "exit": "refused", "error": f"AGENTS.md above the workspace: {upward}"}
    if spec.run_as:
        os.umask(0o007)
    hostfs.install_workspace(job, ws, spec.workspace_files)
    created = hostfs.install_home(job, home, spec.home_files)
    auth_dir = Path(spec.auth.dir).expanduser() if spec.auth else None
    creds = (
        hostfs.link_credentials(home, auth_dir, spec.auth.files) if spec.auth and auth_dir else []
    )
    if spec.run_as:
        hostfs.share_with_group(ws)
        hostfs.share_with_group(home)
    argv = [_sub(a, home, ws) for a in spec.argv]
    env = {k: _sub(v, home, ws) for k, v in spec.env.items()}
    notes_before = (
        hostfs.sha256_file(ws / spec.notes_file) if (ws / spec.notes_file).exists() else None
    )
    report["pre_commit"] = commit_workspace(ws, f"before {spec.mode} {spec.run}")
    cli = cli_identity(argv)
    if spec.expect_version and spec.expect_version not in str(cli["version"]):
        return {
            **report,
            "exit": "refused",
            "error": f"CLI is {cli['version']}, profile fixes {spec.expect_version}",
        }
    mf = hostfs.manifest(
        spec=json.loads(spec.to_json()),
        workspace=ws,
        home=home,
        home_created=created,
        credentials=creds,
        cli=cli,
        env=env,
        secret_env=spec.secret_env,
        host_home=Path.home(),
    )
    hostfs.write_json(out / OUT_MANIFEST, mf)
    watch = [Path(p).expanduser() for p in spec.watch]
    before_watch, before_sessions = hostfs.snapshot(watch), set(codex_session.session_files(home))
    outcome = run_cli(spec, argv, env, ws, home, job)
    session = _session(spec, out / OUT_STDOUT, home, before_sessions)
    notes_after = (
        hostfs.sha256_file(ws / spec.notes_file) if (ws / spec.notes_file).exists() else None
    )
    report.update(
        exit=outcome.exit,
        exit_code=outcome.exit_code,
        wall_s=outcome.wall_s,
        session=session,
        cli_version=cli["version"],
        gpu_guard=outcome.gpu,
        watch_changed=hostfs.snapshot_diff(before_watch, hostfs.snapshot(watch)),
        notes_changed=notes_before != notes_after,
        workspace_extra=hostfs.workspace_extra(ws),
        credentials_refreshed=hostfs.restore_credentials(home, auth_dir, creds) if auth_dir else [],
    )
    report["post_commit"] = commit_workspace(ws, f"after {spec.mode} {spec.run} ({outcome.exit})")
    return {**report, **pack_outputs(ws, home, out, creds, with_git=False)}


def pack_outputs(
    ws: Path, home: Path, out: Path, creds: list[str], with_git: bool
) -> dict[str, Any]:
    home_tar, home_flagged = hostfs.pack(home, exclude_top=creds)
    ws_tar, ws_flagged = hostfs.pack(ws, exclude_top=() if with_git else (".git",))
    (out / OUT_ARCHIVE).write_bytes(home_tar)
    (out / OUT_WORKSPACE).write_bytes(ws_tar)
    return {"secret_members": home_flagged + [f"workspace/{m}" for m in ws_flagged]}


def close(job: Path, spec: LaunchSpec, remove: bool) -> dict[str, Any]:
    """End of a work unit: pack workspace (with git) and home; delete only on ``remove``
    after the control plane confirmed both archives by hash (conventions §7.1)."""
    data = data_root(job)
    ws, home = data / "work" / spec.work_unit, data / "agent-home" / spec.work_unit
    creds = spec.auth.files if spec.auth else []
    report: dict[str, Any] = {"run": spec.run, "mode": spec.mode, "exit": "ok"}
    report.update(pack_outputs(ws, home, job / "out", creds, with_git=True))
    if remove:
        expected = json.loads((job / "confirm.json").read_text())
        got = {
            "home": hostfs.sha256_file(job / "out" / OUT_ARCHIVE),
            "workspace": hostfs.sha256_file(job / "out" / OUT_WORKSPACE),
        }
        if got != expected:
            return {**report, "exit": "refused", "error": f"archives changed since close: {got}"}
        shutil.rmtree(ws)
        shutil.rmtree(home)
        report["removed"] = [str(ws), str(home)]
    return report


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit("usage: python -m kta.agents.launch <job dir>")
    job = Path(args[0]).resolve()
    spec = LaunchSpec.from_json((job / SPEC_FILE).read_bytes())
    (job / "out").mkdir(exist_ok=True)
    try:
        if spec.mode in ("start", "resume"):
            report = step(job, spec)
        else:
            report = close(job, spec, remove=spec.mode == "remove")
    except KtaError as exc:
        report = {"run": spec.run, "mode": spec.mode, "exit": "start_failed", "error": str(exc)}
    hostfs.write_json(job / "out" / OUT_REPORT, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
