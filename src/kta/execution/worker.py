"""Worker side: run jobs, keep the local blob cache, report capabilities.

The control plane reaches a worker through a transport that starts
``python -m kta.execution.worker <command>`` in the worker's deployment of the same
commit (decision D181). Commands read a JSON argument or a blob stream on stdin and
write JSON (or a blob stream) on stdout:

  info  <spec-json>     EnvManifest body, labels, devices (workerprobe)
  has                   stdin: JSON list of sha256 -> stdout: JSON list of missing ones
  put                   stdin: blob stream -> stores verified blobs in the cache
  get   <sha>...        stdout: blob stream of cached blobs
  run                   stdin: {"job_id", "job", "spec", "device"} -> JobResult fields
  serve                 the ``kta worker`` pull daemon (not implemented: NotSupported)

The GPU-exclusive lease is enforced by the Executor that dispatches to this worker
(``kta.execution.executor``); the worker enforces timeouts, process groups and the
GPU health check after a timeout.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from kta.core.context import data_dir_for, parse_env
from kta.core.errors import ExecutionError, KtaError, NotSupported
from kta.core.jobs import ErrorKind, Job
from kta.execution import sandbox
from kta.execution.blobs import WorkerCache, read_frames, write_frames
from kta.execution.workerprobe import WorkerSpec, probe

HEALTH_TIMEOUT_S = 30.0
KILL_GRACE_S = 10.0
# Environment variables passed from the worker process to host jobs; everything else
# comes from Job.env, so a job's environment is reproducible from its record.
PASS_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "USER", "TMPDIR")


@dataclass(frozen=True)
class WorkerDirs:
    root: Path

    @property
    def cache(self) -> WorkerCache:
        return WorkerCache(self.root / "cache")

    def job_dir(self, job_id: str) -> Path:
        return self.root / "jobs" / job_id


def worker_dirs(environ: Mapping[str, str] | None = None) -> WorkerDirs:
    source = os.environ if environ is None else environ
    env = parse_env(source.get("KTA_ENV"))
    return WorkerDirs(data_dir_for(env, source) / "worker")


def _safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ExecutionError("infra", f"input name {name!r} must be a relative path")
    return path


def _substitute(argv: list[str], values: Mapping[str, str]) -> list[str]:
    out: list[str] = []
    for arg in argv:
        for key, value in values.items():
            arg = arg.replace("{" + key + "}", value)
        out.append(arg)
    return out


def _limit_memory(mem_limit_mb: int) -> None:
    limit = mem_limit_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def gpu_healthy(device: str | None, nvidia_smi: str = "nvidia-smi") -> bool:
    """Whether the GPU answers a query within HEALTH_TIMEOUT_S (decision D191)."""
    smi = shutil.which(nvidia_smi)
    if smi is None:
        return False
    argv = [smi, "--query-gpu=index,utilization.gpu", "--format=csv,noheader"]
    if device is not None:
        argv.insert(1, f"--id={device}")
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=HEALTH_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait(timeout=KILL_GRACE_S)


def _classify(returncode: int, timed_out: bool, stderr_tail: bytes) -> ErrorKind:
    if timed_out:
        return "timeout"
    if returncode == 0:
        return "none"
    # SIGKILL that KTA did not send, or an allocator failure, is an out-of-memory kill.
    if returncode in (-signal.SIGKILL, 128 + signal.SIGKILL) or b"MemoryError" in stderr_tail:
        return "oom"
    return "nonzero"


def _prepare(job: Job, job_dir: Path, cache: WorkerCache) -> None:
    (job_dir / "out").mkdir(parents=True, exist_ok=True)
    (job_dir / ".home").mkdir(exist_ok=True)
    for name, sha in job.inputs.items():
        dest = job_dir / _safe_name(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache.path(sha), dest)


def _command(
    job: Job, job_id: str, job_dir: Path, spec: WorkerSpec, device: str | None
) -> tuple[list[str], dict[str, str] | None]:
    """The argv to start and its environment (None: inherited by docker itself)."""
    if job.container is not None:
        docker = shutil.which("docker")
        if docker is None:
            raise NotSupported(f"job needs container {job.container} but docker is absent")
        argv = _substitute(
            job.argv,
            {"python": "python", "cuda_home": sandbox.CONTAINER_CUDA_HOME, "job": "/job"},
        )
        cmd = sandbox.docker_argv(
            docker,
            job.container,
            job_id,
            job_dir,
            argv,
            job.env,
            Path(spec.cuda_home),
            device,
            job.mem_limit_mb,
        )
        return cmd, None
    if job.mem_limit_mb is not None and job.uses_gpu:
        # RLIMIT_AS breaks the CUDA driver's address reservations (decision D190).
        raise NotSupported("memory limits for host GPU jobs; run the job in a container")
    argv = _substitute(
        job.argv, {"python": sys.executable, "cuda_home": spec.cuda_home, "job": str(job_dir)}
    )
    env = {k: os.environ[k] for k in PASS_ENV if k in os.environ}
    env.update(job.env)
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = device
    return argv, env


def run_job(
    job: Job, job_id: str, spec: WorkerSpec, device: str | None, dirs: WorkerDirs
) -> dict[str, Any]:
    cache = dirs.cache
    job_dir = dirs.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=False)
    try:
        _prepare(job, job_dir, cache)
        argv, env = _command(job, job_id, job_dir, spec, device)
        stdout_path, stderr_path = job_dir / ".stdout", job_dir / ".stderr"
        mem = job.mem_limit_mb if job.container is None else None
        start = time.monotonic()
        with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
            proc = subprocess.Popen(
                argv,
                cwd=job_dir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,  # its own process group, killed as a whole
                preexec_fn=(lambda: _limit_memory(mem)) if mem is not None else None,
            )
            timed_out = False
            try:
                proc.wait(timeout=job.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                if job.container is not None and (docker := shutil.which("docker")):
                    subprocess.run(sandbox.kill_argv(docker, job_id), capture_output=True)
                _kill_group(proc)
        wall_s = time.monotonic() - start
        stderr_tail = stderr_path.read_bytes()[-4096:]
        kind = _classify(proc.returncode, timed_out, stderr_tail)
        if kind in ("timeout", "oom") and job.uses_gpu and not gpu_healthy(device, spec.nvidia_smi):
            kind = "gpu_hang"
        outputs = {
            str(p.relative_to(job_dir / "out")): cache.add(p)
            for p in sorted((job_dir / "out").rglob("*"))
            if p.is_file()
        }
        return {
            "ok": kind == "none",
            "exit_code": proc.returncode,
            "wall_s": wall_s,
            "error_kind": kind,
            "stdout_blob": cache.add(stdout_path),
            "stderr_blob": cache.add(stderr_path),
            "outputs": outputs,
            "device": device,
        }
    finally:
        # The job directory is scratch: inputs and outputs live on in the cache (D194).
        shutil.rmtree(job_dir, ignore_errors=True)


def _emit(data: Any) -> None:
    """A reply on stdout: the worker protocol's channel to the control plane."""
    sys.stdout.write(json.dumps(data, sort_keys=True) + "\n")  # kta:allowed protocol reply
    sys.stdout.flush()


def _fail(message: str, code: int) -> int:
    """An error on stderr; the transport maps the exit code to an error type."""
    sys.stderr.write(message + "\n")  # kta:allowed protocol error channel
    return code


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        return _fail("usage: python -m kta.execution.worker info|has|put|get|run|serve", 2)
    cmd, rest = args[0], args[1:]
    try:
        if cmd == "serve":
            raise NotSupported("the pull transport daemon (see kta/execution/transport/pull.md)")
        dirs = worker_dirs()
        if cmd == "info":
            _emit(probe(WorkerSpec.from_dict(json.loads(rest[0]))))
        elif cmd == "has":
            _emit(dirs.cache.missing(json.loads(sys.stdin.read())))
        elif cmd == "put":
            stored = []
            for sha, data in read_frames(sys.stdin.buffer):
                dirs.cache.add_verified(sha, data)
                stored.append(sha)
            _emit(stored)
        elif cmd == "get":
            write_frames(sys.stdout.buffer, dirs.cache.files, rest)
        elif cmd == "run":
            req = json.loads(sys.stdin.read())
            job = Job.from_dict(req["job"])
            spec = WorkerSpec.from_dict(req["spec"])
            _emit(run_job(job, req["job_id"], spec, req.get("device"), dirs))
        else:
            return _fail(f"unknown worker command {cmd!r}", 2)
    except NotSupported as exc:
        return _fail(f"not supported: {exc}", 3)
    except KtaError as exc:
        return _fail(f"worker error: {exc}", 4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
