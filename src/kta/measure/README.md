# kta.measure

Layer 5. Specification: [spec/modules/measure-check.md](../../../spec/modules/measure-check.md). Built by: WP3.

## Responsibility

Measurement harness and versioned measurement protocols; comparisons and profiles, run on GPU workers through the Executor.

## Public interface

- `measure`, `compare`, `profile`

## Dependencies

- May import: `core`, `store`, `targets`, `execution`, `backends`, `knowledge`, `models`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
