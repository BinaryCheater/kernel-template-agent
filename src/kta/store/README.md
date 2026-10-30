# kta.store

Layer 1. Specification: [spec/modules/core-store.md](../../../spec/modules/core-store.md). Built by: WP1.

## Responsibility

Append-only evidence records, content-addressed blobs, operations tables (tasks, locks, pointers, spans, budgets), status derivation, stale propagation and pool snapshots, on SQLite.

## Public interface

- `Store`: put, get, query, search, neighbors, lineage, status, put_blob, blob_path
- `Ops`: enqueue, lease, renew, finish, acquire, release, set_pointer (CAS), reserve, settle, events
- `stale.propagate(id)`, `snapshot(members, parent, note)`

## Dependencies

- May import: `core`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
