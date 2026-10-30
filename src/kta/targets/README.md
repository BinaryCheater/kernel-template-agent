# kta.targets

Layer 2. Specification: [spec/modules/target-exec.md](../../../spec/modules/target-exec.md). Built by: WP2.

## Responsibility

Hardware targets: probing a worker's GPU, driver, CUDA and backend versions through the Executor; unit table; `hw.*` limits (HwFacts); hardware feature list (HwFeatures: ISA instruction families; abstract Capabilities and per-Target declarations live in the knowledge pool, D442; HwFeatures with probed support).

## Public interface

- `TargetPlugin`: probe, units, limits, features, compile_flags; `ProbeReport` (record IDs of a probe)
- `Gb10Target` (`resolve("target", ctx)`): `probe(ex)` writes Target, HwFact and HwFeature records; `latest_target_id()`; `last_probe`
- `gb10/features.yaml`: the open feature list (PTX instruction families with probe kernels)
- Worker-side jobs (run on GPU workers only): `python -m kta.targets.gb10.devinfo`, `python -m kta.targets.gb10.featrun`; `cudadrv.Driver` (ctypes CUDA driver API)

## Dependencies

- May import: `core`, `store`. Probes use the structural `kta.core.jobs.JobRunner`, not `kta.execution`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
