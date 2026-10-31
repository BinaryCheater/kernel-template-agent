"""scripts/worker-setup <worker>: deploy the current commit to a worker and register it.

Steps, all re-runnable:
  1. uv: use the worker's uv if its PATH has one; otherwise install the pinned uv release
     into <root>/.uv/bin (project-local, sha256-checked against the release's checksum).
  2. code: deploy the committed HEAD to <root>/src/<commit> (scripts/remote-run sync).
  3. env: `uv sync --frozen` in that tree (the worker's [workers.<name>].env applies, e.g.
     uv cache and interpreter directories inside the project root).
  4. image: when the worker has an `image` and Docker, build it with docker/build.
  5. register: `kta worker register <name>` on this control plane, which probes the worker
     through the Executor transport and writes its EnvManifest and Worker records.

The worker comes from [workers.<name>] of the configuration in force (KTA_CONFIG, else
KTA_MODE -> config/modes/<mode>.toml, else config/kta.toml; WP23 D600); only ssh workers
are deployed.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tomllib

from kta.core.config import default_config_path
from kta_remote.__main__ import ROOT, Host, _git, _prelude, cmd_sync, ssh

UV_VERSION = "0.11.2"
UV_RELEASES = "https://github.com/astral-sh/uv/releases/download"


def step(msg: str) -> None:
    sys.stderr.write(f"worker-setup: {msg}\n")
    sys.stderr.flush()


def install_uv(host: Host) -> int:
    dest = f"{host.root}/.uv/bin"
    script = f"""set -e
a=$(uname -m); f=uv-$a-unknown-linux-gnu.tar.gz; d="{dest}"
mkdir -p "$d/.download" && cd "$d/.download"
curl -fsSLO {UV_RELEASES}/{UV_VERSION}/$f
curl -fsSLO {UV_RELEASES}/{UV_VERSION}/$f.sha256
sha256sum -c $f.sha256
tar xzf $f
mv uv-$a-unknown-linux-gnu/uv uv-$a-unknown-linux-gnu/uvx "$d/"
cd "$d" && rm -rf .download
"$d/uv" --version"""
    return ssh(host, f"{_prelude(host)} && sh -c {shlex.quote(script)}")


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.exit("usage: scripts/worker-setup <worker>   (a [workers.<name>] in config/kta.toml)")
    name = argv[0]
    cfg_path = default_config_path()
    workers = tomllib.loads(cfg_path.read_text(encoding="utf-8"))["workers"]
    if name not in workers:
        sys.exit(f"worker-setup: no [workers.{name}] in {cfg_path}")
    w = workers[name]
    if w.get("transport") != "ssh":
        sys.exit(
            f"worker-setup: {name} uses transport {w.get('transport')!r}; only ssh is deployed"
        )
    root = w["root"].replace("~", "$HOME", 1) if w["root"].startswith("~") else w["root"]
    host = Host(name, w["ssh"], w["root"], list(w.get("path", [])))
    env = " ".join(
        f"{k}={shlex.quote(v.replace('~', '$HOME', 1))}" for k, v in w.get("env", {}).items()
    )

    dirs = [v for v in w.get("env", {}).values() if v.startswith(w["root"])]
    if dirs and ssh(host, f"{_prelude(host)} && mkdir -p {' '.join(map(shlex.quote, dirs))}") != 0:
        return 1
    step("checking uv")
    if ssh(host, f"{_prelude(host)} && uv --version") != 0:
        step(f"uv missing: installing uv {UV_VERSION} into {w['root']}/.uv/bin")
        if install_uv(host) != 0:
            return 1
    step("deploying committed HEAD")
    if cmd_sync(host) != 0:
        return 1
    commit = _git("rev-parse", "HEAD")
    tree = f'"{root}/src/{commit}"'
    step("uv sync --frozen")
    if ssh(host, f"{_prelude(host)} && cd {tree} && env {env} uv sync --frozen --quiet") != 0:
        return 1
    if w.get("image"):
        image = shlex.quote(w["image"])
        step(f"building image {w['image']} (if Docker is present)")
        script = (
            f"cd {tree} && if command -v docker >/dev/null; then docker/build {image}; "
            "else echo 'no docker: image skipped'; fi"
        )
        if ssh(host, f"{_prelude(host)} && {script}") != 0:
            return 1
    step("registering through the Executor transport")
    environ = dict(os.environ)
    environ.setdefault("KTA_ENV", "dev")
    proc = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(ROOT),
            "--frozen",
            "--quiet",
            "kta",
            "worker",
            "register",
            name,
        ],
        env=environ,
        check=False,
    )
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
