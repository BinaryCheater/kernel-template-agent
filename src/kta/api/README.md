# kta.api

Layer 9. Specification: [spec/modules/api-web.md](../../../spec/modules/api-web.md). Built by: WP9a, WP9b.

## Responsibility

HTTP API (read and operation endpoints) for the web frontend; holds the single store write connection.

## Public interface

- read endpoints (WP9a)
- operation endpoints (WP9b)

## Dependencies

- May import: `core`, `store`, `targets`, `execution`, `backends`, `knowledge`, `models`, `measure`, `formulation`, `checks`, `solve`, `orchestrator`, `agents`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
