# kta.orchestrator

Layer 7. Specification: [spec/modules/orchestrator-cli.md](../../../spec/modules/orchestrator-cli.md). Built by: WP6 (solve, maintain); WP11 (the rest).

## Responsibility

Deterministic loops (solve, expand, learn, calibrate, maintain, experiment, summarize), task queue, budgets, completion checks and pending items.

## Public interface

- `step(loop, ctx) -> Transition`

## Dependencies

- May import: `core`, `store`, `targets`, `execution`, `backends`, `knowledge`, `models`, `measure`, `formulation`, `checks`, `solve`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
