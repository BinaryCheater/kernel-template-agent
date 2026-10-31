"""Quiet-host verification of measurements (WP23, D605).

On a unified-memory host any other activity (a compile, an agent CLI, a solve) shares
memory bandwidth and clocks with the kernel being timed. The worker-side timing job
therefore reads host counters three times: ``host_pre`` before warm-up, ``host_start``
right before the measured loop and ``host_end`` right after it, and stores them raw in
``timing.json``. The control plane recomputes the derived values from the raw counters
and compares them with the thresholds of the controls knob ``measure.quiet_thresholds``
(status default, set by X6). Beyond a threshold the measurement is ``interfered``: it is
kept as a Report, requeued, and never written as a Measurement (``kta.measure.runner``).

Counters (Linux only; the job runs on the worker):

- CPU: ``/proc/stat`` busy and total jiffies, the job process's own CPU seconds
  (``os.times``), so CPU used by *other* processes over the measured loop is
  ``(busy delta / HZ - own delta) / interval`` cores;
- load: ``/proc/loadavg``; memory: ``/proc/meminfo`` MemTotal - MemAvailable;
- GPU: NVML clocks, temperature, power, throttle reasons, utilisation, memory and the
  number of compute processes (the job's own context included).
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from kta.core.errors import NotSupported

PROC = Path("/proc")


class GpuSampler(Protocol):
    def sample(self) -> dict[str, Any]: ...

    def load(self) -> dict[str, Any]: ...


def parse_stat(text: str) -> tuple[int, int]:
    """(busy, total) jiffies of the aggregate ``cpu`` line of /proc/stat."""
    for line in text.splitlines():
        if line.startswith("cpu "):
            v = [int(x) for x in line.split()[1:]]
            idle = v[3] + (v[4] if len(v) > 4 else 0)  # idle + iowait
            total = sum(v[:8])  # guest time is already inside user
            return total - idle, total
    raise NotSupported("/proc/stat has no aggregate cpu line")


def parse_meminfo(text: str) -> dict[str, int]:
    kb = {
        line.split(":")[0]: int(line.split()[1])
        for line in text.splitlines()
        if line.split(":")[0] in ("MemTotal", "MemAvailable")
    }
    if len(kb) != 2:
        raise NotSupported("/proc/meminfo lacks MemTotal or MemAvailable")
    return {
        "mem_total_bytes": kb["MemTotal"] * 1024,
        "mem_used_bytes": (kb["MemTotal"] - kb["MemAvailable"]) * 1024,
    }


def read_counters(gpu: GpuSampler | None, proc: Path = PROC) -> dict[str, Any]:
    """One raw reading of the host (worker side)."""
    try:
        busy, total = parse_stat((proc / "stat").read_text())
        load1 = float((proc / "loadavg").read_text().split()[0])
        mem = parse_meminfo((proc / "meminfo").read_text())
    except OSError as exc:
        raise NotSupported(f"quiet-host counters need /proc: {exc}") from exc
    t = os.times()
    out: dict[str, Any] = {
        "mono_s": time.monotonic(),
        "cpu_busy_jiffies": busy,
        "cpu_total_jiffies": total,
        "hz": os.sysconf("SC_CLK_TCK"),
        "ncpu": os.cpu_count(),
        "own_cpu_s": t.user + t.system + t.children_user + t.children_system,
        "load1": load1,
        **mem,
    }
    if gpu is not None:
        out["gpu"] = {**gpu.sample(), **gpu.load()}
    return out


def other_cpu_cores(start: Mapping[str, Any], end: Mapping[str, Any]) -> float:
    """Cores' worth of CPU used by other processes between two readings."""
    interval = float(end["mono_s"]) - float(start["mono_s"])
    if interval <= 0:
        return 0.0
    busy_s = (int(end["cpu_busy_jiffies"]) - int(start["cpu_busy_jiffies"])) / int(start["hz"])
    own_s = float(end["own_cpu_s"]) - float(start["own_cpu_s"])
    return max(0.0, (busy_s - own_s) / interval)


@dataclass(frozen=True)
class QuietVerdict:
    interfered: bool
    verified: bool  # False when the job recorded no host counters (old protocol jobs)
    reasons: list[str] = field(default_factory=lambda: list[str]())
    observed: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())

    def to_dict(self) -> dict[str, Any]:
        return {
            "interfered": self.interfered,
            "verified": self.verified,
            "reasons": list(self.reasons),
            "observed": dict(self.observed),
        }


def _gpu(reading: Mapping[str, Any]) -> Mapping[str, Any]:
    return reading.get("gpu") or {}


def observed(timing: Mapping[str, Any]) -> dict[str, Any]:
    """The values the thresholds are compared with, recomputed from the raw readings."""
    pre, s, e = timing["host_pre"], timing["host_start"], timing["host_end"]
    gpre, gs, ge = _gpu(pre), _gpu(s), _gpu(e)
    procs = gpre.get("compute_procs")
    sm_max = gs.get("sm_max_mhz")
    sm = [x for x in (gs.get("sm_mhz"), ge.get("sm_mhz")) if x is not None]
    throttle = 0
    for x in (gs.get("throttle_reasons"), ge.get("throttle_reasons")):
        throttle |= int(x or 0)
    temps = [x for x in (gs.get("temp_c"), ge.get("temp_c")) if x is not None]
    return {
        "other_cpu_cores": round(other_cpu_cores(s, e), 4),
        "other_cpu_cores_pre": round(other_cpu_cores(pre, s), 4),
        "load1": max(float(pre["load1"]), float(e["load1"])),
        "mem_used_bytes": max(int(pre["mem_used_bytes"]), int(e["mem_used_bytes"])),
        "gpu_util_pct_pre": gpre.get("gpu_util_pct"),
        # NVML counts the timing job's own context once it exists; before warm-up it has
        # one already (the driver is initialised), so others = count - 1.
        "other_gpu_procs": None if procs is None else max(0, int(procs) - 1),
        "temp_c_max": max(temps) if temps else None,
        "sm_clock_drop_frac": (
            None if not sm or not sm_max else round(1 - min(sm) / float(sm_max), 4)
        ),
        "throttle_reasons": throttle,
    }


def assess(timing: Mapping[str, Any], thresholds: Mapping[str, Any]) -> QuietVerdict:
    """Compare a timing job's host readings with ``measure.quiet_thresholds``."""
    if not all(k in timing for k in ("host_pre", "host_start", "host_end")):
        return QuietVerdict(False, False, ["no host counters recorded"])
    obs = observed(timing)
    reasons: list[str] = []

    def above(name: str, value: Any, limit: Any) -> None:
        if value is not None and limit is not None and float(value) > float(limit):
            reasons.append(f"{name} {value} > {limit}")

    above("other_cpu_cores", obs["other_cpu_cores"], thresholds.get("other_cpu_cores_max"))
    above("other_cpu_cores_pre", obs["other_cpu_cores_pre"], thresholds.get("other_cpu_cores_max"))
    above("load1", obs["load1"], thresholds.get("load1_max"))
    gb = obs["mem_used_bytes"] / 2**30
    above("mem_used_gb", round(gb, 2), thresholds.get("mem_used_gb_max"))
    above("gpu_util_pct_pre", obs["gpu_util_pct_pre"], thresholds.get("gpu_util_pct_pre_max"))
    above("other_gpu_procs", obs["other_gpu_procs"], thresholds.get("other_gpu_procs_max"))
    above("temp_c", obs["temp_c_max"], thresholds.get("temp_c_max"))
    above("sm_clock_drop_frac", obs["sm_clock_drop_frac"], thresholds.get("sm_clock_drop_max"))
    forbidden = int(thresholds.get("throttle_forbidden_mask", 0))
    if obs["throttle_reasons"] & forbidden:
        reasons.append(f"throttle reasons {obs['throttle_reasons']:#x} & {forbidden:#x}")
    return QuietVerdict(bool(reasons), True, reasons, obs)
