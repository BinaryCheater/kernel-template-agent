# kta.targets

Layer 2. Specification: [spec/modules/target-exec.md](../../../spec/modules/target-exec.md). Built by: WP2.

## Responsibility

Hardware targets: probing a worker's GPU, driver, CUDA and backend versions; unit table; `hw.*` limits; hardware feature list.

## Public interface

- `TargetPlugin`: probe, units, limits, features, compile_flags

## Dependencies

- May import: `core`, `store`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
