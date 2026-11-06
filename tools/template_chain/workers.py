"""Remote jobs: dev checks and compile records on one host, timing on another under its lock."""

import fcntl
import json
import shlex
import subprocess  # kta:allowed one experiment harness, same path as tools/kta_source_replay
import tarfile
import time
import uuid
from pathlib import Path

CHECK_HOST = "spark-hk"  # checks need no quiet machine; the timing host stays idle
TIME_HOST = "spark"
ROOTS = {"spark-hk": "kta/e3", "spark": "kta-work/e3"}
LOCKS = {"spark": "kta-work/hostlock/rw.lock"}
WORKER = "kta-worker:tilelang-0.1.15"
BENCH = Path(__file__).with_name("bench.py")
GATE = Path("/tmp/kta-template-chain-check.lock")
# PERFMODEL batches share the timing host and take no flock, so wait them out and notice overlap
BUSY = "pgrep -f '[p]erfmodel/remote/run.py'"


def ssh(host: str, script: str, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", host, "bash", "-s"],
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def upload(host: str, remote: str, family: Path | None, files: dict[str, str]) -> None:
    local = Path(f"/tmp/kta-chain-{uuid.uuid4().hex}.tgz")
    try:
        with tarfile.open(local, "w:gz") as t:
            t.add(BENCH, arcname="bench.py")
            if family:
                for name in ("family.py", "form.py", "form_api.py"):
                    if (family / name).exists():
                        t.add(family / name, arcname=f"family/{name}")
                for path in sorted(family.glob("*.py")):
                    if path.name not in ("family.py", "form.py", "form_api.py"):
                        t.add(path, arcname=f"family/{path.name}")
        done = ssh(host, f"mkdir -p {remote}", 60)
        if done.returncode:
            raise RuntimeError(done.stderr[-400:])
        subprocess.run(
            ["scp", "-q", str(local), f"{host}:{remote}/job.tgz"], check=True, timeout=120
        )
        body = "".join(
            f"cat > {shlex.quote(name)} <<'KTAEOF'\n{text}\nKTAEOF\n"
            for name, text in files.items()
        )
        done = ssh(host, f"cd {remote} && tar xzf job.tgz\n{body}", 60)
        if done.returncode:
            raise RuntimeError(done.stderr[-400:])
    finally:
        local.unlink(missing_ok=True)


def docker(image: str, request: str, result: str, name: str, cache: str) -> str:
    cuda = "-v /usr/local/cuda:/opt/kta-cuda:ro -e CUDA_HOME=/opt/kta-cuda "
    path = "-e PATH=/opt/kta-cuda/bin:/usr/local/bin:/usr/bin:/bin " if image == WORKER else ""
    return (
        f"docker run --rm --name {name} --gpus all --network none --user $(id -u):$(id -g) "
        f'-e HOME=/tmp -e TILELANG_CACHE_DIR=/cache -v "$PWD:/w" -v "{cache}:/cache" '
        f'-v "{cache}/../baseline:/w/baseline:ro" '
        f"{cuda if image == WORKER else ''}{path}-w /w {image} python bench.py {request} {result} "
        f"> {result}.log 2>&1\n"
    )


def fetch(host: str, remote: str, name: str) -> dict:
    done = ssh(host, f"cat {remote}/{name}", 120)
    return json.loads(done.stdout) if done.returncode == 0 and done.stdout.strip() else {}


def compile_jobs(task: str, family: Path, jobs: list[dict], timeout: float = 1800) -> dict:
    """Compile and run without timing. One at a time from this machine."""
    host = CHECK_HOST
    tag = uuid.uuid4().hex[:12]
    remote = f"{ROOTS[host]}/jobs/{tag}"
    request = {
        "task": task,
        "family": "/w/family/family.py",
        "records": True,
        "work": "/w",
        "jobs": jobs,
        "warm": 0,
        "n": 0,
    }
    upload(host, remote, family, {"request.json": json.dumps(request)})
    cache = f"$HOME/{ROOTS[host]}/cache"
    run = f"cd {remote}\nmkdir -p {cache}\n" + docker(
        WORKER, "request.json", "out.json", f"c{tag}", cache
    )
    with GATE.open("a") as gate:
        fcntl.flock(gate, fcntl.LOCK_EX)
        done = ssh(host, f"{run}echo $?", timeout)
    out = fetch(host, remote, "out.json")
    if not out:
        tail = ssh(host, f"tail -c 1500 {remote}/out.json.log", 60).stdout
        return {j["id"]: {"error": f"worker failed: {tail or done.stderr[-400:]}"} for j in jobs}
    return out


def measure(task: dict, name: str, family: Path, shape: dict, jobs: list[dict]) -> dict:
    """Repeat a batch that a PERFMODEL run overlapped or whose comparator moved over 5 %."""
    for attempt in range(4):
        batch = once(task, name, family, shape, jobs)
        before, after = (batch.get(k) or {} for k in ("before", "after"))
        a, b = before.get("median_us"), after.get("median_us")
        batch["valid"] = bool(a and b and not batch["overlap"] and max(a, b) / min(a, b) <= 1.05)
        batch["attempt"] = attempt
        if batch["valid"]:
            break
    return batch


def once(task: dict, name: str, family: Path, shape: dict, jobs: list[dict]) -> dict:
    """One exclusive batch: comparator, candidates, comparator; clocks and load both ends."""
    host = TIME_HOST
    tag = uuid.uuid4().hex[:12]
    remote = f"{ROOTS[host]}/jobs/{tag}"
    base = {"task": name, "warm": 10, "n": 30, "work": "/w"}
    cmp = {"id": "comparator", "kind": "comparator", "timing": True, "shape": shape, "seed": 11}
    files = {
        "cmp.json": json.dumps({**base, "jobs": [cmp]}),
        "cand.json": json.dumps(
            {**base, "family": "/w/family/family.py", "records": True, "jobs": jobs}
        ),
    }
    upload(host, remote, family, files)
    cache = f"$HOME/{ROOTS[host]}/cache"
    image = task["image"]
    clocks = (
        "nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader >> clocks.txt\n"
        "uptime >> clocks.txt\n"
    )
    body = (
        f"cd {remote}\nmkdir -p {cache}\nexec 9>$HOME/{LOCKS[host]}\ndate +%s > queued\n"
        "flock -w 14400 -x 9 || exit 4\n"
        + "while "
        + BUSY
        + " > /dev/null; do sleep 20; done\n"
        + "date +%s > acquired\n"
        + clocks
        + docker(image, "cmp.json", "cmp0.json", f"a{tag}", cache)
        + docker(WORKER, "cand.json", "cand.result.json", f"b{tag}", cache)
        + docker(image, "cmp.json", "cmp1.json", f"d{tag}", cache)
        + clocks
        + f"{BUSY} > overlap.txt || true\n"
    )
    began = time.time()
    done = ssh(host, body, 16000)
    return {
        "before": fetch(host, remote, "cmp0.json").get("comparator"),
        "after": fetch(host, remote, "cmp1.json").get("comparator"),
        "rows": fetch(host, remote, "cand.result.json"),
        "clocks": ssh(host, f"cat {remote}/clocks.txt", 60).stdout,
        "overlap": ssh(host, f"cat {remote}/overlap.txt", 60).stdout.strip(),
        "queue_s": _queue(host, remote),
        "wall_s": time.time() - began,
        "remote": f"{host}:{remote}",
        "exit": done.returncode,
    }


def _queue(host: str, remote: str) -> float | None:
    got = ssh(host, f"cat {remote}/queued {remote}/acquired", 60).stdout.split()
    return float(got[1]) - float(got[0]) if len(got) == 2 else None
