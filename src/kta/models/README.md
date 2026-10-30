# kta.models

Layer 4. Specification: [spec/modules/formulation-solve-models.md](../../../spec/modules/formulation-solve-models.md). Built by: WP5.

## Responsibility

Per-unit cost models and lower bounds; calibration against measurements.

## Public interface

- `bound(case, shape, hw, work)`
- `UnitModel`: predict, fit

## Dependencies

- May import: `core`, `store`, `targets`, `execution`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
