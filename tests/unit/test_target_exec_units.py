"""Pure parts of targets, execution and backends."""

from __future__ import annotations

import hashlib
import io
import signal
import subprocess
from pathlib import Path

import pytest

from kta.backends.base import GeneratedKernel, LaunchSpec, Param
from kta.backends.common import parse_ptxas
from kta.backends.cuda import compile_script
from kta.backends.tilelang import launch_mismatch
from kta.cli.main import main as kta_main
from kta.core.errors import ConfigError, ExecutionError, NotSupported
from kta.core.jobs import Job
from kta.execution.base import WorkerConfig
from kta.execution.blobs import END, frame, parse_frames, read_frames
from kta.execution.registry import label_matches
from kta.execution.sandbox import docker_argv
from kta.execution.transport import make_transport
from kta.execution.transport.pull import PullTransport
from kta.execution.transport.ssh import MAX_CONCURRENT, SshTransport, remote_path
from kta.execution.worker import _classify, _safe_name, _substitute
from kta.targets.gb10 import COMPILE_FLAGS, Gb10Target
from kta.targets.gb10.probe import load_features
from kta.targets.gb10.units import UNITS


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_blob_frames_round_trip() -> None:
    blobs = {sha(b"a"): b"a", sha(b"\x00\n" * 10): b"\x00\n" * 10, sha(b""): b""}
    assert parse_frames(frame(sorted(blobs.items()))) == blobs


def test_blob_frames_refuse_corruption_and_truncation() -> None:
    good = frame([(sha(b"hello"), b"hello")])
    with pytest.raises(ExecutionError, match="does not match"):
        parse_frames(good.replace(b"hello", b"hellO"))
    with pytest.raises(ExecutionError):
        list(read_frames(io.BytesIO(good[: -len(END) - 2])))


def test_label_matching() -> None:
    labels = {"gpu": "sm_121", "pool.gpu-shared": "2", "backend.tilelang": "0.1.15"}
    assert label_matches(labels, {"gpu": "sm_121", "pool": "gpu-shared", "backend": "tilelang"})
    assert not label_matches(labels, {"gpu": "sm_90"})
    assert not label_matches(labels, {"pool": "cpu"})
    assert label_matches(labels, {})


def test_default_pools_by_kind() -> None:
    assert Job(kind="compile", argv=["x"]).pool == "cpu"
    assert Job(kind="run", argv=["x"]).pool == "gpu-shared"
    assert Job(kind="measure", argv=["x"]).pool == "gpu-exclusive"
    assert Job(kind="shell", argv=["x"], requires={"pool": "gpu-shared"}).uses_gpu


def test_worker_config_validation() -> None:
    cfg = WorkerConfig.from_mapping(
        "w", {"transport": "ssh", "ssh": "w", "root": "~/k", "pools": {"cpu": "4"}}
    )
    assert cfg.pools == {"cpu": 4} and cfg.spec.pools == {"cpu": 4}
    with pytest.raises(ConfigError, match="unknown keys"):
        WorkerConfig.from_mapping("w", {"transport": "ssh", "hosts": "x"})
    with pytest.raises(ConfigError, match="transport"):
        WorkerConfig.from_mapping("w", {"transport": "carrier-pigeon"})


def test_ssh_script_runs_the_matching_deployment(tmp_path: Path) -> None:
    cfg = WorkerConfig.from_mapping(
        "spark",
        {
            "transport": "ssh",
            "ssh": "spark",
            "root": "~/kta-work",
            "path": ["~/.local/bin"],
            "env": {"UV_CACHE_DIR": "/x y"},
        },
    )
    t = SshTransport(cfg, "dev", "WP2", "abc123", tmp_path / "ssh")
    script = t.script(["info", '{"a": 1}'])
    assert '"$HOME"/kta-work/src/abc123/.venv/bin/python' in script
    assert "KTA_ENV=dev" in script and "KTA_WP=WP2" in script and "UV_CACHE_DIR='/x y'" in script
    assert "exit 97" in script and "kta.execution.worker info" in script
    argv = t.argv(["has"])
    assert argv[0] == "ssh" and "BatchMode=yes" in argv and argv[-2] == "spark"
    assert remote_path("~") == '"$HOME"' and remote_path("/a b") == "'/a b'"


def test_ssh_cancel_is_not_held_behind_running_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = WorkerConfig.from_mapping("spark", {"transport": "ssh", "ssh": "spark", "root": "~/k"})
    t = SshTransport(cfg, "dev", None, "abc123", tmp_path / "ssh")
    assert "kta.execution.worker cancel 01JOB" in t.script(["cancel", "01JOB"])

    def fake_run(argv: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 0, b'{"stopped": true}', b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    for _ in range(MAX_CONCURRENT):  # every slot held, as by long ``run`` calls
        t.slots.acquire()
    assert t.call(["cancel", "01JOB"], b"", 1) == b'{"stopped": true}'
    assert not t.slots.acquire(blocking=False)  # other commands still wait for a slot


def test_pull_transport_is_not_supported(tmp_path: Path) -> None:
    cfg = WorkerConfig.from_mapping("far", {"transport": "pull"})
    t = make_transport(cfg, "dev", "WP2", tmp_path, "abc")
    assert isinstance(t, PullTransport)
    with pytest.raises(NotSupported, match=r"pull\.md"):
        t.call(["info"], b"", 1)


def test_worker_daemon_is_not_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KTA_ENV", "test")
    assert kta_main(["worker", "serve"]) == 3


def test_worker_helpers() -> None:
    assert str(_safe_name("in/a.bin")) == "in/a.bin"
    for bad in ("../x", "/etc/passwd", ""):
        with pytest.raises(ExecutionError):
            _safe_name(bad)
    assert _substitute(
        ["{python}", "-m", "{cuda_home}/bin"], {"python": "py", "cuda_home": "/c"}
    ) == [
        "py",
        "-m",
        "/c/bin",
    ]
    assert _classify(0, False, b"") == "none"
    assert _classify(-signal.SIGKILL, True, b"") == "timeout"
    assert _classify(-signal.SIGKILL, False, b"") == "oom"
    assert _classify(1, False, b"MemoryError") == "oom"
    assert _classify(2, False, b"") == "nonzero"


def test_container_command(tmp_path: Path) -> None:
    argv = docker_argv(
        "docker",
        "sha256:abc",
        "01ABC",
        tmp_path,
        ["sh", "build.sh"],
        {"K": "v"},
        Path("/usr/local/cuda"),
        None,
        512,
    )
    joined = " ".join(argv)
    assert "--network none" in joined and "--memory 512m" in joined
    assert "/usr/local/cuda:/opt/kta-cuda:ro" in joined and "--gpus" not in joined
    assert argv[-3:] == ["sha256:abc", "sh", "build.sh"]


PTXAS = """\
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'other' for 'sm_121a'
ptxas info    : Used 40 registers, used 0 barriers, 352 bytes cmem[0]
ptxas info    : Compiling entry function 'vadd' for 'sm_121a'
ptxas info    : Function properties for vadd
    0 bytes stack frame, 8 bytes spill stores, 4 bytes spill loads
ptxas info    : Used 10 registers, used 0 barriers, 1024 bytes smem, 380 bytes cmem[0]
"""


def test_ptxas_feedback_for_the_entry() -> None:
    fb = parse_ptxas(PTXAS, "vadd")
    assert (fb["regs"], fb["smem"]) == (10, 1024)
    assert parse_ptxas(PTXAS, "other")["regs"] == 40
    assert parse_ptxas(PTXAS, "missing") == {"ptxas": []}


def test_ptxas_spills_line_is_counted() -> None:
    text = PTXAS.replace("    0 bytes stack frame", "ptxas info    : 0 bytes stack frame")
    assert parse_ptxas(text, "vadd")["spills"] == 12


def test_tilelang_launch_must_match() -> None:
    launch = LaunchSpec([4096], [256], 0, [Param("A", "f32*"), Param("B", "f32*")])
    gk = GeneratedKernel("tilelang", {"kernel.py": ""}, "f", launch)
    assert launch_mismatch(gk, {"block": 256, "params": ["A", "B"]}) is None
    assert "threads per block" in str(launch_mismatch(gk, {"block": 128, "params": ["A", "B"]}))
    assert "parameters" in str(launch_mismatch(gk, {"block": 256, "params": ["B", "A"]}))


def test_launch_spec_round_trip() -> None:
    launch = LaunchSpec([2, 3], [32, 4], 128, [Param("x", "f16*"), Param("n", "i32")])
    assert LaunchSpec.from_dict(launch.to_dict()) == launch
    assert launch.threads_per_block == 128


def test_cuda_compile_script_uses_target_flags() -> None:
    script = compile_script(COMPILE_FLAGS["cuda"])
    assert script.startswith("{cuda_home}/bin/nvcc -cubin -arch=sm_121a")
    assert "cuobjdump -sass out/module.cubin" in script


def test_features_yaml_is_well_formed() -> None:
    feats = load_features()
    assert len(feats) >= 40
    assert all(f.unit in UNITS for f in feats)
    assert all("kta_probe" in f.source and f.instructions for f in feats)
    assert {f.mode for f in feats} == {"run", "compile"}


def test_gb10_flags_and_unknown_backend() -> None:
    plugin = Gb10Target.__new__(Gb10Target)
    assert "-arch=sm_121a" in plugin.compile_flags("tilelang")
    with pytest.raises(NotSupported):
        plugin.compile_flags("triton")
