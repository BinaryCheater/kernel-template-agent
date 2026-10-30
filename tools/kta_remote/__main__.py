"""scripts/remote-run: manual ssh operations on construction hosts.

  exec   <host> -- <cmd...>        run in the host's project root, stream output, keep exit code
  bg     <host> <name> -- <cmd...> start in the background; output in <root>/runs/<name>/out.log
  status <host> <name>             running / exited <code>
  logs   <host> <name> [-n N]      tail the background run's log
  sync   <host>                    deploy the committed HEAD to <root>/src/<commit>, link current

Hosts and their roots come from config/remote.toml. Every ssh call uses BatchMode so it
never prompts. ``sync`` refuses a dirty tree: runs from uncommitted code are not evidence.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "remote.toml"
RUN_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


@dataclass(frozen=True)
class Host:
    name: str
    ssh: str
    root: str
    path: list[str]


def load_host(name: str) -> Host:
    hosts = tomllib.loads(CONFIG.read_text(encoding="utf-8")).get("hosts", {})
    if name not in hosts:
        sys.exit(f"remote-run: unknown host {name!r}; known: {', '.join(sorted(hosts))}")
    h = hosts[name]
    return Host(name, h["ssh"], h["root"], list(h.get("path", [])))


def _root(host: Host) -> str:
    # "~" must expand on the remote side, so it is left unquoted.
    return host.root.replace("~", "$HOME", 1) if host.root.startswith("~") else host.root


def _prelude(host: Host) -> str:
    path = ":".join(p.replace("~", "$HOME", 1) for p in host.path)
    root = _root(host)
    export = f'export PATH="{path}:$PATH"; ' if path else ""
    return f'{export}mkdir -p "{root}" && cd "{root}"'


def ssh(host: Host, script: str, stdin: bytes | None = None) -> int:
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host.ssh, script]
    return subprocess.run(argv, input=stdin, check=False).returncode


def _check_name(name: str) -> str:
    if not name or not set(name) <= RUN_NAME_CHARS:
        sys.exit("remote-run: run names use letters, digits, '-', '_' and '.' only")
    return name


def cmd_exec(host: Host, command: list[str]) -> int:
    return ssh(host, f"{_prelude(host)} && {shlex.join(command)}")


def cmd_bg(host: Host, name: str, command: list[str]) -> int:
    run = f"runs/{_check_name(name)}"
    inner = f"{shlex.join(command)}; echo $? > {run}/exit"
    script = (
        f"{_prelude(host)} && mkdir -p {run} && rm -f {run}/exit || exit 1; "
        f"nohup sh -c {shlex.quote(inner)} > {run}/out.log 2>&1 < /dev/null & "
        f"echo $! > {run}/pid; echo started {name} pid $(cat {run}/pid)"
    )
    return ssh(host, script)


def cmd_status(host: Host, name: str) -> int:
    run = f"runs/{_check_name(name)}"
    script = (
        f"{_prelude(host)} && if [ -f {run}/exit ]; then echo exited $(cat {run}/exit); "
        f"elif [ -f {run}/pid ] && kill -0 $(cat {run}/pid) 2>/dev/null; then echo running; "
        f"else echo unknown; exit 1; fi"
    )
    return ssh(host, script)


def cmd_logs(host: Host, name: str, lines: int) -> int:
    run = f"runs/{_check_name(name)}"
    return ssh(host, f"{_prelude(host)} && tail -n {int(lines)} {run}/out.log")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def cmd_sync(host: Host) -> int:
    if _git("status", "--porcelain"):
        sys.exit("remote-run: working tree is dirty; commit first (dirty runs are not evidence)")
    commit = _git("rev-parse", "HEAD")
    archive = subprocess.run(
        ["git", "-C", str(ROOT), "archive", "--format=tar", commit],
        capture_output=True,
        check=True,
    ).stdout
    dest = f"src/{commit}"
    script = (
        f"{_prelude(host)} && mkdir -p {dest} && tar -x -C {dest} && "
        f"echo {commit} > {dest}/.kta-commit && ln -sfn {dest} current && "
        f"echo synced {commit[:12]} to {host.name}:{host.root}/{dest}"
    )
    return ssh(host, script, stdin=archive)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripts/remote-run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("exec")
    p.add_argument("host")
    p.add_argument("command", nargs=argparse.REMAINDER)
    p = sub.add_parser("bg")
    p.add_argument("host")
    p.add_argument("name")
    p.add_argument("command", nargs=argparse.REMAINDER)
    for name in ("status", "logs"):
        p = sub.add_parser(name)
        p.add_argument("host")
        p.add_argument("name")
        if name == "logs":
            p.add_argument("-n", type=int, default=50)
    p = sub.add_parser("sync")
    p.add_argument("host")
    args = parser.parse_args(argv)
    host = load_host(args.host)
    command: list[str] = list(getattr(args, "command", []))
    if command[:1] == ["--"]:
        command = command[1:]
    if args.cmd in {"exec", "bg"} and not command:
        parser.error("missing command after --")
    if args.cmd == "exec":
        return cmd_exec(host, command)
    if args.cmd == "bg":
        return cmd_bg(host, args.name, command)
    if args.cmd == "status":
        return cmd_status(host, args.name)
    if args.cmd == "logs":
        return cmd_logs(host, args.name, args.n)
    return cmd_sync(host)


if __name__ == "__main__":
    sys.exit(main())
