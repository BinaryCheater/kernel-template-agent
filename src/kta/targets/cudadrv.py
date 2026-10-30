"""Minimal CUDA driver API binding (ctypes), used only inside worker jobs.

Both backends produce cubins; every module is loaded and launched through this one
path (the harness, decision D186), so launch overhead and timing are the same for all
backends. Nothing here runs on the control plane.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence

from kta.core.errors import ExecutionError, NotSupported

CUresult = ctypes.c_int
CUdeviceptr = ctypes.c_uint64

# CUdevice_attribute values, checked against cuda.h of CUDA 13.0.
ATTRIBUTES: dict[str, int] = {
    "max_threads_per_block": 1,
    "max_block_dim_x": 2,
    "max_grid_dim_x": 5,
    "max_shared_memory_per_block": 8,
    "total_constant_memory": 9,
    "warp_size": 10,
    "max_registers_per_block": 12,
    "clock_rate_khz": 13,
    "multiprocessor_count": 16,
    "memory_clock_rate_khz": 36,
    "global_memory_bus_width": 37,
    "l2_cache_size": 38,
    "max_threads_per_multiprocessor": 39,
    "compute_capability_major": 75,
    "compute_capability_minor": 76,
    "max_shared_memory_per_multiprocessor": 81,
    "max_registers_per_multiprocessor": 82,
    "max_shared_memory_per_block_optin": 97,
    "max_blocks_per_multiprocessor": 106,
    "max_persisting_l2_cache_size": 108,
    "reserved_shared_memory_per_block": 111,
    "cluster_launch": 120,
}
FUNC_ATTRIBUTES = {"shared_size_bytes": 1, "local_size_bytes": 3, "num_regs": 4}
LIBRARY_NAMES = ("libcuda.so.1", "libcuda.so")


def _load() -> ctypes.CDLL:
    for name in LIBRARY_NAMES:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise NotSupported("no CUDA driver library (libcuda.so.1) on this host")


class Driver:
    """One device and its primary context."""

    def __init__(self, device: int = 0) -> None:
        self.lib = _load()
        self._check(self.lib.cuInit(0), "cuInit")
        self.device = ctypes.c_int()
        self._check(self.lib.cuDeviceGet(ctypes.byref(self.device), device), "cuDeviceGet")
        self.ctx = ctypes.c_void_p()
        self._check(
            self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(self.ctx), self.device),
            "cuDevicePrimaryCtxRetain",
        )
        self._check(self.lib.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")

    def _check(self, code: int, what: str) -> None:
        if code != 0:
            name = ctypes.c_char_p()
            self.lib.cuGetErrorName(code, ctypes.byref(name))
            label = name.value.decode() if name.value else str(code)
            raise ExecutionError("nonzero", f"{what} failed: {label}")

    # ---- device queries ----------------------------------------------------------

    def attribute(self, attr: int) -> int:
        value = ctypes.c_int()
        self._check(
            self.lib.cuDeviceGetAttribute(ctypes.byref(value), attr, self.device),
            f"cuDeviceGetAttribute({attr})",
        )
        return value.value

    def attributes(self) -> dict[str, int]:
        return {name: self.attribute(code) for name, code in ATTRIBUTES.items()}

    def name(self) -> str:
        buf = ctypes.create_string_buffer(256)
        self._check(self.lib.cuDeviceGetName(buf, 256, self.device), "cuDeviceGetName")
        return buf.value.decode()

    def total_mem(self) -> int:
        size = ctypes.c_size_t()
        self._check(
            self.lib.cuDeviceTotalMem_v2(ctypes.byref(size), self.device), "cuDeviceTotalMem"
        )
        return size.value

    def driver_version(self) -> int:
        v = ctypes.c_int()
        self._check(self.lib.cuDriverGetVersion(ctypes.byref(v)), "cuDriverGetVersion")
        return v.value

    # ---- modules and launches ----------------------------------------------------

    def load_function(self, cubin: bytes, entry: str) -> ctypes.c_void_p:
        module = ctypes.c_void_p()
        image = ctypes.create_string_buffer(cubin)
        self._check(self.lib.cuModuleLoadData(ctypes.byref(module), image), "cuModuleLoadData")
        func = ctypes.c_void_p()
        self._check(
            self.lib.cuModuleGetFunction(ctypes.byref(func), module, entry.encode()),
            f"cuModuleGetFunction({entry})",
        )
        return func

    def func_attribute(self, func: ctypes.c_void_p, attr: int) -> int:
        value = ctypes.c_int()
        self._check(
            self.lib.cuFuncGetAttribute(ctypes.byref(value), attr, func), "cuFuncGetAttribute"
        )
        return value.value

    def set_dynamic_smem(self, func: ctypes.c_void_p, nbytes: int) -> None:
        max_dynamic = 8  # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        self._check(
            self.lib.cuFuncSetAttribute(func, max_dynamic, nbytes), "cuFuncSetAttribute(smem)"
        )

    def alloc(self, nbytes: int) -> CUdeviceptr:
        ptr = CUdeviceptr()
        self._check(
            self.lib.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(max(nbytes, 1))), "cuMemAlloc"
        )
        return ptr

    def free(self, ptr: CUdeviceptr) -> None:
        self._check(self.lib.cuMemFree_v2(ptr), "cuMemFree")

    def to_device(self, ptr: CUdeviceptr, data: bytes) -> None:
        self._check(self.lib.cuMemcpyHtoD_v2(ptr, data, ctypes.c_size_t(len(data))), "cuMemcpyHtoD")

    def to_host(self, ptr: CUdeviceptr, nbytes: int) -> bytes:
        buf = ctypes.create_string_buffer(nbytes)
        self._check(self.lib.cuMemcpyDtoH_v2(buf, ptr, ctypes.c_size_t(nbytes)), "cuMemcpyDtoH")
        return buf.raw

    def memset(self, ptr: CUdeviceptr, value: int, nbytes: int) -> None:
        self._check(
            self.lib.cuMemsetD8_v2(ptr, ctypes.c_ubyte(value), ctypes.c_size_t(nbytes)),
            "cuMemsetD8",
        )

    def launch(
        self,
        func: ctypes.c_void_p,
        grid: Sequence[int],
        block: Sequence[int],
        smem: int,
        args: Sequence[ctypes._SimpleCData],  # pyright: ignore[reportMissingTypeArgument]
    ) -> None:
        g = [*grid, 1, 1][:3]
        b = [*block, 1, 1][:3]
        params = (ctypes.c_void_p * max(len(args), 1))()
        for i, a in enumerate(args):
            params[i] = ctypes.cast(ctypes.byref(a), ctypes.c_void_p)
        self._check(
            self.lib.cuLaunchKernel(
                func, *[ctypes.c_uint(x) for x in (*g, *b, smem)], None, params, None
            ),
            "cuLaunchKernel",
        )

    def synchronize(self) -> None:
        self._check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")

    def event(self) -> ctypes.c_void_p:
        ev = ctypes.c_void_p()
        self._check(self.lib.cuEventCreate(ctypes.byref(ev), 0), "cuEventCreate")
        return ev

    def record(self, ev: ctypes.c_void_p) -> None:
        self._check(self.lib.cuEventRecord(ev, None), "cuEventRecord")

    def elapsed_ms(self, start: ctypes.c_void_p, end: ctypes.c_void_p) -> float:
        self._check(self.lib.cuEventSynchronize(end), "cuEventSynchronize")
        ms = ctypes.c_float()
        self._check(self.lib.cuEventElapsedTime(ctypes.byref(ms), start, end), "cuEventElapsed")
        return float(ms.value)
