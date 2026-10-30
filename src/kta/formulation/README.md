# kta.formulation

Layer 5. Specification: [spec/modules/formulation-solve-models.md](../../../spec/modules/formulation-solve-models.md). Built by: WP5.

## Responsibility

Assemble a formulation (variables, constraints, cost blocks) from a chain's items.

## Public interface

- `Formulation`: family variables, constraints (each with its source: chain item key or `glue`), cost terms (with source), `linear`, `cost_linear`, per-item `blocks`, `reused`, `hw_refs`, `case_refs`, `missing_hw`, `fingerprint`.
- `assemble(family, chain, templates, case, hw, previous=None)`: one block per chain item (`blocks.build_block`), plus glue; unchanged blocks are taken from `previous`.
- `per_unit_work(f, assignment, shape, hw) -> (overlappable, serial)` work per unit (D240).
- Diagnostics: `violated(...)` (which constraints exclude a point), `cost_breakdown(...)` (each cost term's work with its source).
- Binding semantics (D246): family variable, constant, list-domain value, or expression over family variables / `case.*` / `hw.*`; `p.*` bindings likewise; `p.instances` scales the item's cost terms (D247). Preconditions without `same_as` are constraints (D253).

## Dependencies

- May import: `core`, `store`, `targets`, `execution`, `backends`, `knowledge`, `models`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
