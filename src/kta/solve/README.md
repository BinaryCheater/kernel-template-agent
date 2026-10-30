# kta.solve

Layer 6. Specification: [spec/modules/formulation-solve-models.md](../../../spec/modules/formulation-solve-models.md). Built by: WP5.

## Responsibility

Enumeration and MIP solving of formulations; re-ranking with build information.

## Public interface

- `solve(f, case, shape_bucket, model, hw, ...)`
- `rerank(candidates, builds, model, hw, k)`

## Dependencies

- May import: `core`, `store`, `targets`, `execution`, `backends`, `knowledge`, `models`, `measure`, `formulation`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
