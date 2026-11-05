"""Retained direct remote experiment jobs with live verified GPU queue events."""

from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import math
import shlex
import tarfile
import time
from pathlib import Path
from typing import Any

from kta_template_replay.remote import invoke


def remaining_after_queue(limit_s: float, elapsed_s: float, excluded_s: float) -> int:
    if not all(math.isfinite(x) and x >= 0 for x in (limit_s, elapsed_s, excluded_s)):
        raise ValueError("Budget inputs must be finite and nonnegative")
    return max(0, int(limit_s - max(0, elapsed_s - excluded_s)))


def event(path: Path, request: dict, name: str, at_s: float, **evidence: Any) -> None:
    row = {
        "request_id": request["request_id"],
        "event": name,
        "at_s": at_s,
        "slot": request["slot"],
        "worker": "spark-hk",
        "job_id": request["request_id"],
        "submission_id": request["submission_id"],
    }
    row.update(evidence)
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def execute(
    root: Path,
    attempt: Path,
    request: dict,
    campaign: dict,
    baseline: dict,
    timeout_s: float,
    ledger: Path | None,
) -> dict[str, Any]:
    # Queue on the controller instead of holding nine SSH channels while flock waits.
    # The resource is still the same GPU, protected again by the remote host locks.
    import fcntl

    data = attempt
    while data.name != "source-attention-control-v1":
        data = data.parent
        if data == data.parent:
            raise ValueError("Cannot locate campaign root")
    queued_at = time.time()
    queued = False
    queue_request = {**request, "request_id": request["request_id"] + "-controller"}
    with (data / "gpu-dispatch.lock").open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if ledger and not queued:
                    event(ledger, queue_request, "blocked", queued_at)
                    event(
                        ledger,
                        queue_request,
                        "queued",
                        queued_at,
                        queue_kind="controller-gpu-dispatch",
                    )
                    queued = True
                if ledger:
                    event(
                        ledger,
                        queue_request,
                        "waiting",
                        time.time(),
                        queue_kind="controller-gpu-dispatch",
                    )
                time.sleep(0.25)
        waited = time.time() - queued_at
        if ledger and queued:
            event(
                ledger, queue_request, "acquired", time.time(), queue_kind="controller-gpu-dispatch"
            )
            event(ledger, queue_request, "completed", time.time())
        result = execute_remote(root, attempt, request, campaign, baseline, timeout_s, ledger)
        result["controller_queue_s"] = waited
        result["queue_s"] = result.get("queue_s", 0) + waited
        result["wall_s"] = result.get("wall_s", 0) + waited
        return result


def execute_remote(
    root: Path,
    attempt: Path,
    request: dict,
    campaign: dict,
    baseline: dict,
    timeout_s: float,
    ledger: Path | None,
) -> dict[str, Any]:
    request_began = time.monotonic()
    if ledger:
        event(ledger, request, "blocked", time.time())
    host = baseline.get("host", "spark-hk")
    remote = f"data/dev/source-attention-control-v1/slots/{request['slot']}/{request['request_id']}"
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as stream:
        for path in attempt.iterdir():
            if path.name in {"worker.py", "request.json", "campaign.json", "candidate.py"}:
                stream.add(path, arcname=path.name, recursive=False)
    upload = invoke(
        root,
        host,
        f"mkdir -p {shlex.quote(remote)} && printf %s "
        f"{shlex.quote(base64.b64encode(archive.getvalue()).decode())} | base64 -d | "
        f"tar xz -C {shlex.quote(remote)}",
    )
    (attempt / "upload.stderr").write_bytes(upload.stderr)
    if upload.returncode:
        if ledger:
            event(ledger, request, "completed", time.time())
        return {"external_failure": True, "reason": "transport upload failed", "active_s": 0.0}
    image = shlex.quote(campaign["infrastructure"]["base_container"])
    source = shlex.quote(baseline["remote_path"])
    # This monitors host PIDs during the entire window, rather than inferring exclusivity
    # from pre/post empty process lists. The only permitted process is this container PID.
    script = f"""set -u
work="$PWD/{remote}"
cache="$PWD/data/dev/source-attention-control-v1/slots/{request["slot"]}/cache"
mkdir -p "$work/.home" "$cache" "$work/out"
exec 8>"$HOME/kta/hostlock/intent.lock"
exec 9>"$HOME/kta/hostlock/rw.lock"
date +%s.%N > "$work/queued"
flock -w 3600 -x 8 || exit 4
flock -w 3600 -x 9 || exit 4
date +%s.%N > "$work/acquired"
waited=0
while test ! -f "$work/launch-budget.txt"; do
  sleep 0.05
  waited=$((waited+1))
  test "$waited" -lt 1200 || exit 4
done
read deadline < "$work/launch-budget.txt"
nvidia-smi -q > "$work/environment-before.txt"
uptime > "$work/load-before.txt"
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits > "$work/processes-before.txt"
name={shlex.quote("source-" + request["request_id"].lower())}
budget=$(awk -v deadline="$deadline" -v now="$(date +%s.%N)" \\
 'BEGIN {{ r=deadline-now; if(r<0)r=0; printf "%.3f", int(r*1000)/1000 }}')
printf '%s\\n' "$budget" > "$work/effective-worker-budget.txt"
if ! awk -v r="$budget" 'BEGIN {{ exit !(r>0) }}'; then
  printf '124\\n' > "$work/exit.txt"
  date +%s.%N > "$work/completed"
  exit 0
fi
timeout --signal=TERM --kill-after=10 "$budget" docker run --rm --name "$name" \\
 --network none --gpus all --user "$(id -u):$(id -g)" \\
 -v "$work:/work" -v {source}:/baseline:ro -v "$cache:/cache" \\
 -v /usr/local/cuda:/opt/kta-cuda:ro -w /work \\
 -e HOME=/work/.home -e USER=kta -e XDG_CACHE_HOME=/cache/xdg -e CUDA_HOME=/opt/kta-cuda \\
 -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton \\
 -e CUDA_CACHE_PATH=/cache/cuda -e TVM_CACHE_DIR=/cache/tvm \\
 -e PATH=/opt/kta-cuda/bin:/usr/local/bin:/usr/bin:/bin \\
 {image} python worker.py > "$work/stdout.txt" 2> "$work/stderr.txt" &
runner=$!
while kill -0 "$runner" 2>/dev/null; do
  stamp=$(date +%s.%N)
  permitted=$(docker inspect --format '{{{{.State.Pid}}}}' "$name" 2>/dev/null || true)
  printf 'sample %s allowed %s\\n' "$stamp" "$permitted" >> "$work/process-monitor.txt"
  container_id=$(docker inspect --format '{{{{.Id}}}}' "$name" 2>/dev/null || true)
  for gpu_pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits); do
    printf '%s\\n' "$gpu_pid" >> "$work/process-monitor.txt"
    membership=$(cat "/proc/$gpu_pid/cgroup" 2>/dev/null || true)
    if test -n "$container_id" && printf '%s' "$membership" | grep -Fq "$container_id"; then
      printf 'sample %s allowed %s\\n' "$stamp" "$gpu_pid" >> "$work/process-monitor.txt"
      printf '%s %s %s %s\\n' "$stamp" "$gpu_pid" "$container_id" "$membership" \\
        >> "$work/container-pid-proof.txt"
    fi
  done
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw \\
    --format=csv,noheader >> "$work/telemetry.txt"
  sleep 0.25
done
wait "$runner"
printf '%s\\n' "$?" > "$work/exit.txt"
date +%s.%N > "$work/completed"
nvidia-smi -q > "$work/environment-after.txt"
uptime > "$work/load-after.txt"
exit 0
"""
    (attempt / "remote-command.sh").write_text(script)
    observed: dict[str, float] = {}
    budget_sent = False
    began = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(invoke, root, host, script)
        while not future.done():
            markers = invoke(
                root,
                host,
                f"for n in queued acquired; do "
                f'test ! -f {shlex.quote(remote)}/"$n" || '
                f'printf \'%s %s\\n\' "$n" "$(cat {shlex.quote(remote)}/"$n")"; done; '
                f"if test -f {shlex.quote(remote)}/queued && "
                f"test ! -f {shlex.quote(remote)}/acquired; "
                "then printf 'waiting %s\\n' \"$(date +%s.%N)\"; fi",
            )
            publish_markers(markers.stdout, ledger, request, observed)
            if "acquired" in observed and not budget_sent:
                excluded = max(0, observed["acquired"] - observed["queued"])
                if ledger:
                    excluded = max(0, observed["acquired_local"] - observed["queued_local"])
                elapsed_at_handoff = time.monotonic() - request_began
                remaining = remaining_after_queue(timeout_s, elapsed_at_handoff, excluded)
                deadline = observed["acquired"] + remaining
                with (attempt / "budget-handoff.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "at_s": time.time(),
                                "limit_s": timeout_s,
                                "elapsed_s": elapsed_at_handoff,
                                "excluded_queue_s": excluded,
                                "worker_timeout_s": remaining,
                                "remote_acquired_at_s": observed["acquired"],
                                "conservative_deadline_remote_s": deadline,
                            }
                        )
                        + "\n"
                    )
                handoff = invoke(
                    root,
                    host,
                    f"printf '%s\\n' {deadline:.6f} > {shlex.quote(remote)}/launch-budget.txt",
                )
                (attempt / "budget-handoff.stderr").write_bytes(handoff.stderr)
                budget_sent = handoff.returncode == 0
            try:
                future.result(timeout=1)
            except concurrent.futures.TimeoutError:
                continue
        result = future.result()
    wall_s = time.monotonic() - began
    (attempt / "transport.stderr").write_bytes(result.stderr)
    fetched = invoke(
        root,
        host,
        f"cd {shlex.quote(remote)} && tar czf - "
        "queued acquired completed launch-budget.txt effective-worker-budget.txt "
        "exit.txt stdout.txt stderr.txt "
        "environment-before.txt environment-after.txt load-before.txt load-after.txt "
        "processes-before.txt process-monitor.txt container-pid-proof.txt telemetry.txt out "
        "2>/dev/null | base64 -w0",
    )
    output = attempt / "remote-output"
    output.mkdir()
    data = base64.b64decode(fetched.stdout)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as stream:
        stream.extractall(output, filter="data")
    marker_bytes = b"".join(
        name.encode() + b" " + (output / name).read_bytes()
        for name in ("queued", "acquired")
        if (output / name).exists()
    )
    publish_markers(marker_bytes, ledger, request, observed)
    if ledger:
        event(ledger, request, "completed", time.time())
    raw_queue_s = max(
        0, observed.get("acquired", time.time()) - observed.get("queued", time.time())
    )
    queue_s = (
        max(0, observed["acquired_local"] - observed["queued_local"])
        if ledger and "acquired_local" in observed
        else raw_queue_s
    )
    receipt: dict[str, Any] = {
        "remote_path": remote,
        "wall_s": wall_s,
        "queue_s": queue_s,
        "active_s": max(0, wall_s - queue_s),
        "verified_remote_queue_s": raw_queue_s,
    }
    path = output / "out/result.json"
    if not path.exists():
        return {
            **receipt,
            "external_failure": result.returncode not in {0, 124, 137},
            "passed": False,
            "reason": "worker omitted result; inspect exit/stderr",
        }
    worker = json.loads(path.read_text())
    competing = competing_processes(output)
    worker["competing_gpu_processes"] = competing
    if competing:
        worker["window_valid"] = False
        worker["meets_target"] = False
    return {
        **receipt,
        "worker": worker,
        "external_failure": (worker.get("passed", False) and not worker.get("window_valid", False)),
    }


def publish_markers(raw: bytes, ledger: Path | None, request: dict, observed: dict) -> None:
    for line in raw.decode().splitlines():
        name, value = line.split()
        if name == "waiting" and "queued_local" in observed and "acquired" not in observed:
            now = time.time()
            verified = min(
                now, observed["queued_local"] + max(0, float(value) - observed["queued"])
            )
            if ledger:
                event(
                    ledger,
                    request,
                    "waiting",
                    verified,
                    raw_remote_at_s=float(value),
                    observed_at_s=now,
                )
            continue
        if name not in observed:
            observed[name] = float(value)
            now = time.time()
            local_at = now
            if name == "acquired" and "queued_local" in observed:
                # Both raw timestamps use the same worker clock. Cap the excluded
                # local interval by this verified duration, never gifting transport.
                local_at = min(
                    now,
                    observed["queued_local"] + max(0, observed["acquired"] - observed["queued"]),
                )
            observed[name + "_local"] = local_at
            if ledger:
                event(
                    ledger,
                    request,
                    name,
                    local_at,
                    raw_remote_at_s=float(value),
                    observed_at_s=now,
                    clock_domain="local observed; conservative queue cap",
                )


def competing_processes(output: Path) -> list[str]:
    others = set((output / "processes-before.txt").read_text().split())
    allowed: set[str] = set()
    sampled: set[str] = set()
    for line in (output / "process-monitor.txt").read_text().splitlines():
        if line.startswith("sample "):
            parts = line.split()
            if len(parts) == 4 and parts[3] != "0":
                allowed.add(parts[3])
        elif line.strip():
            sampled.add(line.strip())
    return sorted(others | (sampled - allowed))
