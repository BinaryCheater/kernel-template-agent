# WP1 delivery: core and store

Branch `wp/WP1`, cut from `integration` @ `955edac` (WP0 merged). Code commit for all evidence runs: `2e8373176cab32d5f211bed8a09a215eb29a8084` (clean). Evidence outputs are kept outside git under `~/kta-data/dev/WP1/evidence/2e8373176cab32d5f211bed8a09a215eb29a8084/` on the control host.

## What was done

Real runs (evidence claims 0 and 1 in `deliveries/WP1.evidence.json`; records in `~/kta-data/dev/WP1/kta.db`):

- **Store benchmark on spark (real run, claim 0).** `kta store bench` on spark's data directory (`KTA_ENV=dev`, `~/kta-work/data/dev/WP1/kta.db`) wrote 100 000 synthetic records (origin `fixture`, D156) in 9.7 s and timed 500 calls per query type. p95 in ms: get 0.013, query by name 0.017, filtered recent 1.033, level range 1.135, in-neighbors 0.018, lineage 0.017, 1-word search 37.116, 2-word search 27.001, kind-filtered search 37.333. All below 50 ms. Stored as Report `2effebf3…f967` (origin `real`, actor commit `2e83731`, dirty false). It references spark's worker EnvManifest `c676a7fd…9e06` (Ubuntu 24.04.5, aarch64, CUDA 13.0.3, driver 580.95.05). The raw latencies are blob `f830c862…c64f`. Output: `store-bench.spark.txt`.
- **Spark records reach the control-plane store (real run, claim 1).** The bundle was copied from spark (same sha256 `a5732b54…3bce` on both hosts) and imported with `kta store import` from a clean tree. The import recomputed both record IDs and the blob hash before writing (D153). `kta doctor` in dev on the control host selects `store`/`ops` = `sqlite [kta.store]`, with no fixtures (`doctor.control.txt`).

Implemented and covered by unit and acceptance tests (no separate real run needed beyond the above):

- **core/schemas/**: a pydantic model for every kind in spec/schemas.md §4, plus Source, Protocol, ExperimentRun and Report (D143). `KINDS` maps each kind to its model, schema version, required refs and ref target kinds (D155). The `Record` envelope normalizes the body (defaults filled, `extra="forbid"`) and `created_at` (UTC) (D144). `Actor` has `commit`/`dirty` (D84, D140). `validate_body(kind, body)` raises `InvalidBody`, which is both a `SchemaError` and a `ValueError` (D142), as the WP10a gate check expects.
- **core/ids.py**: `canonical_json` (sorted keys, compact, UTF-8, floats via repr, UTC datetimes, NaN refused), `record_id`, `UlidFactory` (D145). A cross-process test checks a fixed ID for a record with Chinese text and floats.
- **core/expr.py**: an `ast`-based interpreter (no `eval`). It allows only the operators and functions in schemas.md §3, plus `hw.`/`p.`/`case.` attributes. It checks free variables and computes `is_linear` (add/sub, multiplication by a constant, comparisons other than `!=`, `and`). Exponents are bounded.
- **core/trace.py**: `span()`, `tracing()`, `new_tracer()`, `current_span_id()`, `child_environ()`. `KTA_TRACE_PARENT=<trace>:<span>` carries the parent. A span that raises ends `fail`, an interrupt ends `cancelled` (D148). The Ops `spans` table is the sink.
- **store/**: `SqliteStore` and `SqliteOps` on one SQLite file per environment data dir, registered through `register_implementations` (D77).
  - Records are append-only: triggers refuse UPDATE/DELETE on `records` and `edges`.
  - `put`/`put_many` enforce the environment origin rules, refs existence and kinds, stored blobs, and the status rules (D146, D149).
  - Edges are built from refs, plus `subject` for StatusChanges.
  - FTS5 search uses the per-kind field table and bm25 `rank` (D151). `query` uses the D147 filter syntax.
  - Also: `neighbors`, `lineage`, `status` derivation, and a content-addressed blob store under `objects/`.
  - `stale.propagate` / `stale.reject` (D150) and `snapshot.snapshot` (PoolSnapshot + CAS on `current_pool`).
  - `SqliteOps` covers tasks (lease, renew, finish), locks, CAS pointers, budgets, workers, loops, span sink and queries, and the `ops_events` log (D152).
  - `transfer` provides verified bundles (D153). `bench` is the store benchmark (D156).
- **CLI**: `kta store bench|export|import` (D157).
- **Gate**: the WP1-specific checks in `tools/kta_gate/specific.py` (D154), unit-tested in `tests/unit/test_gate_wp1.py`.
  - The named WP1 acceptance tests must pass.
  - The dev store must hold a real `store-benchmark` Report that is listed in the evidence manifest. It must come from a clean WP1 commit in this branch and reference a worker EnvManifest, with at least 100 000 records. Every p95 must be < 50 ms, recomputed from the raw latency blob.
- **Acceptance tests** (`tests/acceptance/WP1/`), one per acceptance item:
  - `main` rejects fixture/simulated records and dirty writes.
  - The registry refuses a fixture store in dev.
  - IDs are stable across processes.
  - Triggers refuse UPDATE/DELETE.
  - `put` fails on missing refs.
  - The status rules, one parametrized case each.
  - Stale propagation on a five-level graph.
  - With 8 contenders on separate connections, exactly one CAS succeeds.

## Fixtures and simplifications

- `src/kta/store/bench.py` (`# kta:fixture` on `_body`): synthetic record bodies for the store benchmark, as the spec requires ("写入 10 万条合成记录（标 origin="fixture"）"). They are stored with origin `fixture` and are never used outside the benchmark. The timings and the Report are real.
- No other fixtures. The work package allows no simplifications, and none were used. No tests are skipped.

## Known issues

- **Search has the least headroom.** Single-word searches for very common terms match almost every indexed document, and bm25 must score all of them (p95 37 ms at 100 000 records on spark). FTS5's bm25 normalizes by whole-document length, not per field. Much larger stores would need stop-word handling or candidate limits. Watch this in M2 (docs/design/20 §7).
- **Single-writer API not yet built.** api-web.md requires all production writes to go through the API process (WP9a). Today each process opens its own connection, and concurrent writers are serialized by SQLite's file lock. `kta store bench/import` write directly; they are construction and deployment tools.
- **Stale triggers are not automatic.** Deciding when a new Target, HwModel, Contract or checker version supersedes an old one is left to the orchestrator (WP6). The store provides `stale.propagate` and `stale.reject`.
- **No knowledge-relation edges.** The relation edges of docs/design/23 §3 (uses, implements, fits) are not derived. Edges come from refs and StatusChange subjects only (WP4 can add them).
- **Retention and backup are not implemented.** There is no retention job for `ops_events`, and `prune_spans` exists without a scheduler. The daily SQLite backup to slai (docs/design/20 §8) is also missing. Neither is in WP1's list.
- **Status order depends on clocks.** Status derivation orders StatusChanges by `created_at`, then insertion order, so clock skew between hosts could reorder changes.
- **No Worker record yet.** The benchmark Report references an EnvManifest only. Worker registration and job records come with WP2, so Report is not treated as a job record.
- **IDs before `put`.** Callers that compute a record ID before `put` must set `actor.commit`/`dirty` themselves, because the store fills them only when they are absent (D140).
- **`Context` has no `store`/`ops` fields yet.** Code obtains them with `resolve("store", ctx)` / `resolve("ops", ctx)`.
- **New data directory on spark.** `~/kta-work/data/dev/WP1/` (≈137 MB database plus the bundle). Nothing was installed outside project directories.

## New decisions

D140 (needs user review), D141, D142, D143 (needs user review), D144, D145, D146, D147, D148, D149 (needs user review), D150, D151, D152, D153 (needs user review), D154 (needs user review), D155, D156, D157.

## Needed changes

- **Spec edits made in this WP, pending user review where marked.**
  - spec/schemas.md: Actor `commit`/`dirty`, the `schema_` alias, and normalization (D140, D141, D144); §4.10 fields for Source/Protocol/ExperimentRun/Report (D143); the status-rule interpretation (D149).
  - spec/modules/core-store.md: the added files, methods and `tasks.card_type` column (D146, D152).
- **WP2:** replace the manual bundle copy (D153) with Executor result transfer. Job records (Build/Check/Measurement/Profile) must carry `refs.env` and `refs.worker`; `put` enforces this (D155).
- **WP9a:** hold the single production writer (`open_store`/`open_ops`), and route CLI writes through the API.

## Gate output

Run: `KTA_DATA=~/kta-data scripts/gate WP1` on the control host, after the evidence runs.

```text
gate WP1: PASSED (22 pass, 0 warn, 0 fail)
  PASS ruff format
  PASS ruff check
  PASS import layers
  PASS residue
  PASS test doubles
  PASS error handling
  PASS print
  PASS subprocess outside Executor
  PASS secrets
  PASS acceptance lock — 1 locked files unchanged
  PASS pyright
  PASS unit tests — 157 passed
  PASS contract tests — 2 passed
  PASS fake coverage — 1 fake(s) covered
  PASS registry (kta doctor, dev)
  PASS delivery note
  PASS fixture declarations — 1 declared
  PASS decision log — 110 decisions
  PASS skipped tests — none
  PASS evidence — 2 claim(s)
  PASS WP1 acceptance tests — 12 tests passed
  PASS store benchmark record — Report 2effebf38d1c (listed in the evidence manifest)
```
