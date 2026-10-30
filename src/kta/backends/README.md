# kta.backends

Layer 4. Specification: [spec/modules/target-exec.md](../../../spec/modules/target-exec.md). Built by: WP2.

## Responsibility

Compile generated kernels (TileLang, CUDA C++) into cubins through the Executor, with an input-hash build cache and Build records; load and launch every cubin the same way (harness).

## Public interface

- `GeneratedKernel`, `LaunchSpec`, `Param`, `BuiltModule`, `Backend` protocol
- `CudaBackend` (`resolve("backend.cuda", ctx)`), `TileLangBackend` (`resolve("backend.tilelang", ctx)`): supports, build
- `harness.run_job(...)`, `harness.collect(...)`, `harness.Buffer`; worker side `python -m kta.backends.harness.run`
- `smoke.Smoke`: vector add per backend, lease deferral and timeout recovery self-tests (`kta backend smoke|selftest`)

## Dependencies

- May import: `core`, `store`, `targets`, `execution`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
