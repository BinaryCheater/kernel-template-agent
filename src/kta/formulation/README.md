# kta.formulation

Layer 5. Specification: [spec/modules/formulation-solve-models.md](../../../spec/modules/formulation-solve-models.md). Built by: WP5.

## Responsibility

Assemble a formulation (variables, constraints, cost blocks) from a chain's items.

## Public interface

- `Formulation`
- `assemble(family, chain, templates, case, hw)`
- `per_unit_work(f, assignment, shape, hw)`

## Dependencies

- May import: `core`, `store`, `targets`, `execution`, `backends`, `knowledge`, `models`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
