"""Minimal NVML binding (ctypes) for clock, temperature and power samples in worker jobs.

The measurement protocol records the GPU state at the start and end of every timed
run. Values a device does not report (GB10 has no separate memory clock) are ``None``
with the NVML return code kept next to them, so a missing reading stays visible.
"""

from __future__ import annotations

import ctypes
import os
from typing import Any

from kta.core.errors import NotSupported

NVML_CLOCK_SM = 1
NVML_CLOCK_MEM = 2
NVML_TEMPERATURE_GPU = 0
LIBRARY_NAMES = ("libnvidia-ml.so.1", "libnvidia-ml.so")


def _load() -> ctypes.CDLL:
    for name in LIBRARY_NAMES:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise NotSupported("no NVML library (libnvidia-ml.so.1) on this host")


def visible_device() -> int:
    """NVML ignores CUDA_VISIBLE_DEVICES; the worker sets it to the leased device."""
    return int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or 0)


class Nvml:
    def __init__(self, index: int) -> None:
        self.lib = _load()
        code = self.lib.nvmlInit_v2()
        if code != 0:
            raise NotSupported(f"nvmlInit failed with code {code}")
        self.handle = ctypes.c_void_p()
        code = self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.handle))
        if code != 0:
            raise NotSupported(f"nvmlDeviceGetHandleByIndex({index}) failed with code {code}")

    def _uint(self, fn: str, *args: int) -> tuple[int | None, int]:
        value = ctypes.c_uint()
        code = getattr(self.lib, fn)(self.handle, *args, ctypes.byref(value))
        return (value.value if code == 0 else None), code

    def sample(self) -> dict[str, Any]:
        sm, sm_rc = self._uint("nvmlDeviceGetClockInfo", NVML_CLOCK_SM)
        mem, mem_rc = self._uint("nvmlDeviceGetClockInfo", NVML_CLOCK_MEM)
        sm_max, _ = self._uint("nvmlDeviceGetMaxClockInfo", NVML_CLOCK_SM)
        temp, temp_rc = self._uint("nvmlDeviceGetTemperature", NVML_TEMPERATURE_GPU)
        power, power_rc = self._uint("nvmlDeviceGetPowerUsage")
        reasons = ctypes.c_ulonglong()
        r_rc = self.lib.nvmlDeviceGetCurrentClocksThrottleReasons(
            self.handle, ctypes.byref(reasons)
        )
        return {
            "sm_mhz": sm,
            "sm_max_mhz": sm_max,
            "mem_mhz": mem,
            "temp_c": temp,
            "power_w": power / 1000 if power is not None else None,
            "throttle_reasons": reasons.value if r_rc == 0 else None,
            "nvml_codes": {"sm": sm_rc, "mem": mem_rc, "temp": temp_rc, "power": power_rc},
        }

    def close(self) -> None:
        self.lib.nvmlShutdown()
