# kta.store

Layer 1. Specification: [spec/modules/core-store.md](../../../spec/modules/core-store.md). Built by: WP1.

## Responsibility

Append-only evidence records, content-addressed blobs, operations tables (tasks, locks, pointers, spans, budgets), status derivation, stale propagation and pool snapshots, on SQLite.

## Public interface

- `resolve("store", ctx)` / `resolve("ops", ctx)` (registered as `sqlite`), or `open_store(ctx)` / `open_ops(ctx)`; database `<data_dir>/kta.db`, blobs `<data_dir>/objects/`
- `Store` (`SqliteStore`): put, put_many, get, get_json, exists, kind_of, count, query (filter syntax D147), search (FTS5, D151), neighbors, lineage, status, put_blob, blob_path, blob_media
- `Ops` (`SqliteOps`): enqueue, lease, renew, finish, task, update_task / set_task_priority / list_tasks / list_loops (added by WP6, D265, D367), acquire, release, get_pointer, set_pointer (CAS), set_budget, reserve, settle, budget, heartbeat, put_loop, get_loop, events, start_span, end_span, spans, prune_spans
- `status.check_change`, `status.fold`; `stale.propagate(store, id, reason, now)`, `stale.reject(...)`; `snapshot.snapshot(store, ops, members, parent, note, actor, now)`
- `transfer.export_bundle` / `import_bundle` (verified record bundles, D153); `bench.run` (store benchmark, D156)

## Controls versions (WP21)

- `controls.py`: `current` (read only), `ensure` (writes version 1 or a definitions upgrade), `propose` (a human's version with a reason and the version read), `history`. Pointer `controls:current`; records of kind ControlsVersion.

## Dependencies

- May import: `core`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).

## Demotion (WP14, D448)

`status.py`: actor `system` may move a record down the promotion ladder (never up); `stale.propagate` is what the knowledge promotion engine calls after a demotion.

## Backups and hand-over (WP23, D601)

`backup.py`: `backup` (SQLite online backup, blobs, system configuration, manifest; `--handover` marks the store), `restore` (empty data directory, hash-checked), `check_owner` (the API and processes refuse a handed-over store). CLI `kta store backup|restore`.

WP24 (D634): `assurance.py` `effective(store, id)`: declared / tested from StatusChanges, bounded / proved only from the newest live whole-subject ProofSummary; `proof_summary(store, id)`.
