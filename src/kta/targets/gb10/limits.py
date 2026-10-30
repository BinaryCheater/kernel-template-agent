"""``hw.*`` limit names and where their values come from.

Values are HwFact records: device attributes queried through the CUDA driver API on the
GPU worker (method ``documented``), later complemented by microbenchmarks (``measured``).
Two limits are not device attributes; they are documented constants of the CUDA
programming guide's compute-capability table and are written with that source.
"""

from __future__ import annotations

# hw.* name -> CUDA device attribute (kta.targets.cudadrv.ATTRIBUTES key).
FROM_ATTRIBUTES: dict[str, str] = {
    "sm_count": "multiprocessor_count",
    "warp_size": "warp_size",
    "max_threads_per_cta": "max_threads_per_block",
    "max_threads_per_sm": "max_threads_per_multiprocessor",
    "max_ctas_per_sm": "max_blocks_per_multiprocessor",
    "smem_per_cta": "max_shared_memory_per_block_optin",
    "smem_per_cta_default": "max_shared_memory_per_block",
    "smem_per_sm": "max_shared_memory_per_multiprocessor",
    "smem_reserved_per_cta": "reserved_shared_memory_per_block",
    "regs_per_cta": "max_registers_per_block",
    "regs_per_sm": "max_registers_per_multiprocessor",
    "l2_bytes": "l2_cache_size",
    "l2_persisting_bytes": "max_persisting_l2_cache_size",
    "sm_clock_khz": "clock_rate_khz",
    "mem_clock_khz": "memory_clock_rate_khz",
    "mem_bus_bits": "global_memory_bus_width",
    "max_grid_x": "max_grid_dim_x",
    "cluster_launch": "cluster_launch",
}
# Not queryable as device attributes (CUDA C++ Programming Guide, "Technical
# Specifications per Compute Capability").
DOCUMENTED_CONSTANTS: dict[str, int] = {
    "regs_per_thread": 255,
    "max_cluster_size_portable": 8,
}
DOCUMENTED_SOURCE = "CUDA C++ Programming Guide, technical specifications per compute capability"
