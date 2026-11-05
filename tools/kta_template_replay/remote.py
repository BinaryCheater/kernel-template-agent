"""Authorized construction-only direct runner; does not create product APIs.

The runbook/user explicitly permits temporary direct remote experiment commands.
Every shell invocation uses scripts/remote-run, with retained snapshots and locks.
"""

from __future__ import annotations

import base64
import io
import shlex
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any

from .direct import collect, stage
from .manifest import load, write_new


def invoke(root: Path, host: str, script: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [str(root / "scripts/remote-run"), "exec", host, "--", "bash", "-c", script],
        capture_output=True,
        check=False,
    )


def remote_evaluate(
    root: Path, run: Path, candidate: Path, agent_active_s: float, host: str
) -> dict[str, Any]:
    attempt = stage(run, candidate, agent_active_s)
    manifest = load(attempt / "manifest.json")
    remaining_s = int(load(attempt / "reservation.json")["remaining_s"])
    remote = "data/dev/template-replay/evaluations/" + identity_name(attempt)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as stream:
        for name in ("worker.py", "manifest.json", "candidate.py"):
            stream.add(attempt / name, arcname=name, recursive=False)
    encoded = base64.b64encode(archive.getvalue()).decode()
    upload = invoke(
        root,
        host,
        f"mkdir -p {shlex.quote(remote)} && "
        f"printf %s {shlex.quote(encoded)} | base64 -d | "
        f"tar xz -C {shlex.quote(remote)}",
    )
    (attempt / "upload.stderr").write_bytes(upload.stderr)
    if upload.returncode:
        raise RuntimeError("Remote payload upload failed; see retained upload.stderr")
    image = shlex.quote(manifest["environment"]["container"])
    # Exclusive lock is acquired after intent, matching the shared worker lock protocol.
    script = f"""set -u
work="$PWD/{remote}"
mkdir -p "$HOME/kta/hostlock" "$work/.home"
exec 8>"$HOME/kta/hostlock/intent.lock"
exec 9>"$HOME/kta/hostlock/rw.lock"
queued=$(date +%s.%N)
flock -w 3600 -x 8 || exit 4
flock -w 3600 -x 9 || exit 4
started=$(date +%s.%N)
printf '%s %s\\n' "$queued" "$started" > "$work/queue.txt"
nvidia-smi -q > "$work/environment-before.txt"
timeout {max(1, remaining_s)} docker run --rm --network none --gpus all \\
  --user "$(id -u):$(id -g)" \\
  -e HOME=/work/.home -e CUDA_HOME=/opt/kta-cuda \
  -e PATH=/opt/kta-cuda/bin:/usr/local/bin:/usr/bin:/bin \\
  -v /usr/local/cuda:/opt/kta-cuda:ro -v "$work:/work" -w /work \\
  {image} python worker.py > "$work/stdout.txt" 2> "$work/stderr.txt"
code=$?
printf '%s\\n' "$code" > "$work/exit.txt"
nvidia-smi -q > "$work/environment-after.txt"
exit 0
"""
    write_new(
        attempt / "remote.json",
        {
            "host": host,
            "directory": remote,
            "container": manifest["environment"]["container"],
            "script": script,
            "transport": "manual remote commands authorized by user",
        },
    )
    start = time.monotonic()
    execution = invoke(root, host, script)
    elapsed = time.monotonic() - start
    (attempt / "transport.stdout").write_bytes(execution.stdout)
    (attempt / "transport.stderr").write_bytes(execution.stderr)
    fetch_script = (
        f"cd {shlex.quote(remote)} && "
        "tar czf - queue.txt exit.txt stdout.txt stderr.txt "
        "environment-before.txt environment-after.txt out 2>/dev/null | base64 -w0"
    )
    fetched = invoke(root, host, fetch_script)
    (attempt / "fetch.stderr").write_bytes(fetched.stderr)
    data = base64.b64decode(fetched.stdout)
    output = attempt / "remote-output"
    output.mkdir()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as stream:
        stream.extractall(output, filter="data")
    queue_start, acquired = map(float, (output / "queue.txt").read_text().split())
    queue_s = acquired - queue_start
    worker_result = output / "out/result.json"
    if worker_result.exists():
        return collect(attempt, worker_result, elapsed, min(elapsed, max(0, queue_s)))
    # Timeout/transport failures still consume an attempt and retain the candidate.
    result = {
        "passed": False,
        "reason": "worker did not produce result.json",
        "remote_directory": remote,
        "attempt": str(attempt),
        "exit_code": (output / "exit.txt").read_text().strip(),
    }
    write_new(
        attempt / "execution.json",
        {
            "wall_s": elapsed - queue_s,
            "queue_s": queue_s,
            "transport_exit_code": execution.returncode,
        },
    )
    write_new(attempt / "result.json", result)
    return result


def identity_name(attempt: Path) -> str:
    from kta.core.context import SystemClock
    from kta.core.ids import UlidFactory

    return UlidFactory(SystemClock()).new() + "-" + attempt.name
