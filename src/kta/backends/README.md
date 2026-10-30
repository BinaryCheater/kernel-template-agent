# kta.backends

Layer 4. Specification: [spec/modules/target-exec.md](../../../spec/modules/target-exec.md). Built by: WP2.

## Responsibility

Compile generated kernels (TileLang, CUDA C++) into uniform built modules through the Executor.

## Public interface

- `GeneratedKernel`, `BuiltModule`
- `Backend`: supports, build

## Dependencies

- May import: `core`, `store`, `targets`, `execution`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
