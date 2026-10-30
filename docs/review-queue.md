# Review queue

Decisions made without the user that need review. Newest last. Each entry: question, choice made, rationale, cost of changing it later, link to the decision ID.

(empty)

## D77 — Implementation registry is an object on the Context

- **Question:** spec/conventions.md §3.2 gave module-level `register(...)` / `resolve(...)`, which needs a process-wide mutable table, while §3.1 forbids module-level mutable state.
- **Choice:** `Registry` object carried as `ctx.registry`; each package under `src/kta/` exposes `register_implementations(registry)`; `build_registry()` calls those hooks only. `resolve(interface, ctx)` keeps its signature. Selection: the `[impl]` name, else the only real implementation. Spec §3.2 updated.
- **Rationale:** no global state; tests build their own registry; only `src/kta` code reaches a production registry, and `register` refuses fixture flags that contradict where the factory is defined.
- **Cost of changing later:** low now (no implementations registered yet); grows with every WP that registers implementations.

## D84 — Which WPs need record-backed evidence, and how the gate reads records

- **Question:** conventions §5 says the gate checks record IDs remotely, but WP0, WP4–WP6, WP10a and WP13 produce no store records of their own, and the Actor schema does not name the commit/dirty fields.
- **Choice:** record IDs are mandatory for WP1, WP2, WP3, WP7, WP8, WP9a, WP9b, WP10b, WP11, WP12; the gate opens `$KTA_DATA/dev/<WP>/kta.db` on the control host (where the dev store lives, per conventions §5) and requires `actor.commit` == the claim's commit and `actor.dirty` == false; GPU claims need `refs.worker` pointing at a Worker whose `labels` contain `gpu`. Other WPs may list claims with empty `record_ids` (still from a clean, existing commit).
- **Rationale:** matches where the dev store actually is; gives WP1 a concrete field contract.
- **Cost of changing later:** low until WP1 fixes the Actor fields; afterwards a schema migration.

## D95 — Sample data in the frontend before the store exists

- **Question:** you asked for a decent frontend quickly, but real records only exist after WP1 (store) and real runs after WP7.
- **Choice:** WP9a phase A builds the full UI against a small sample dataset kept only in `web/`, flagged in code and shown with a SAMPLE DATA banner on every page; phase B switches to the real API over the store and deletes the sample path. Final WP9a acceptance still requires WP7's real records.
- **Rationale:** lets you review layout and readability now instead of after M1.
- **Cost of changing later:** low; the alternative is to wait for WP1 (frontend design review slips accordingly).

## D140 — Who sets the code provenance (actor.commit, actor.dirty) of a record

- **Question:** core-store.md says the store writes the git commit and dirty flag into every record's actor; D84 fixed the field names. Records produced on a worker are written by the control plane later, so the store's own code state is not always the producer's.
- **Choice:** both fields optional in the model and set together; `Store.put` fills them from its own code state when absent and keeps them when present; `main` refuses `dirty=true`.
- **Rationale:** worker records keep the provenance of the code that actually produced them (conventions §5 requires worker and control plane at the same commit, so normally both agree); IDs stay content hashes of what is stored.
- **Cost of changing later:** medium: making the store always overwrite would change IDs of imported records; refusing mismatched commits is a small check in `put`.

## D143 — Fields for Source, Protocol, ExperimentRun, Report

- **Question:** docs/design/20 §2.3 lists these kinds; spec/schemas.md gives no fields. WP1 must provide models for all kinds.
- **Choice:** minimal bodies (spec/schemas.md §4.10): Source (name, type, locator, revision, license?, note?), Protocol (name, version, type, description, params), ExperimentRun (name, condition, seed?, params, note), Report (title, subject, summary, data).
- **Rationale:** enough for WP1's benchmark Report and for WP3/WP4/WP10a/WP12 to start; open `params`/`data` dicts avoid guessing details.
- **Cost of changing later:** low before WP3/WP4/WP12 write such records; afterwards a schema version bump.

## D149 — Interpretation of the status promotion table

- **Question:** schemas.md §4.9 allows checkers only `proposed -> reproduced/calibrated/holdout_validated`, but re-verification of `stale` objects and stepwise promotion (reproduced -> calibrated) are needed; `-> proposed` by any actor would let an agent undo a human rejection.
- **Choice:** checkers promote from proposed, stale, or a lower promotion level; `-> proposed` by anyone only while proposed, by a human from any status; assurance only upward by checkers; stale propagation does not overwrite rejected/deprecated.
- **Rationale:** keeps human verdicts authoritative and lets re-verification restore a stale object.
- **Cost of changing later:** low: one function (`kta.store.status`) and its tests; existing StatusChange records stay valid unless rules get stricter.

## D153 — Moving spark-produced records into the control-plane store (before the Executor)

- **Question:** the WP1 benchmark must run on spark's data directory, but the gate reads evidence records from `~/kta-data/dev/WP1/kta.db` on the control host (D84), and there is no Executor until WP2.
- **Choice:** run `kta store bench` on spark via `scripts/remote-run`; export the Report with its refs and blobs as a JSON-lines bundle (`kta store bench --bundle`); copy it to the control host; `kta store import` recomputes every ID and blob hash before writing. Records keep the spark actor.
- **Rationale:** content-hash IDs make the transfer verifiable without trusting the copy; the same bundle format can serve WP2 until job results flow through the Executor.
- **Cost of changing later:** low: WP2 may replace the manual copy with Executor result transfer; bundles stay importable.

## D154 — What the WP1 gate checks instead of a GPU acceptance run

- **Question:** spec/gate.md asks WP1–WP3 to run `pytest -m gpu tests/acceptance/<WP>/` via the Executor on spark; WP1 has no GPU work and no Executor.
- **Choice:** require the named WP1 acceptance tests to pass (they run with the unit tests) and a valid real `store-benchmark` Report in the dev store (clean WP1 commit, worker EnvManifest, ≥100 000 records, every p95 < 50 ms recomputed from the raw latency blob, listed in the evidence manifest).
- **Rationale:** checks the same thing the spec's real-run acceptance asks for, from stored evidence rather than stdout.
- **Cost of changing later:** low: one function in `tools/kta_gate/specific.py`.

## D100 — Seed reference files are bodies; refs implied by layout

- **Question:** WP10a must ship YAML that "parses with the pydantic models", but no seed file format existed (WP4 not started) and refs are content-hash IDs that do not exist for seed files.
- **Choice:** one body per file, kind from file / directory name (compatible with the gate's D87 rules), refs implied by the directory layout; Template and Pattern directories are `templates/` and `patterns/`.
- **Rationale:** same convention as family directories; each file validates directly against its model; `kta init` (WP4) can derive refs when importing.
- **Cost of changing later:** low; a mechanical rewrite of 7 directories.

## D101 — Source has no schema

- **Question:** Reference refs `source` and D59 lists Source as a node, but spec/schemas.md defines no Source fields.
- **Choice:** `source.yaml` with repo, commit, commit_date, license, paths, retrieved_at, note.
- **Rationale:** the minimum needed to freeze and re-fetch an upstream source.
- **Cost of changing later:** low; needs a spec/schemas.md addition (listed under "Needed changes" in deliveries/WP10a.md).

## D102 — How match rate and unmatched transactions are counted

- **Question:** schemas.md defines `matched` / `total` and `unmatched_transactions` but not how Templates proposed during a decomposition count, or how partial matches and step-less transactions are represented. These numbers feed the coverage page (21 §2).
- **Choice:** fixed decomposition order defines the pool; only Templates that existed before the reference count as matched; a partial match is not matched; each unmatched step or step-less transaction is one entry with exactly one Gap.
- **Rationale:** gives a discovery curve that depends only on the order and makes the gate's one-to-one check meaningful.
- **Cost of changing later:** low for data (recount from chain.yaml); medium once the coverage page shows these numbers.

## D103 — Variable names of the five seed Templates were assumed

- **Question:** chains must bind seed Template variables before WP4 defines them.
- **Choice:** names from paper.md §3.3 (num_groups, group_size, assignment; level, buffer, share_order; mechanism, issue_point, complete_point, granularity_bytes; S, P, beta; binding, queue_depth).
- **Rationale:** closest authoritative source; keeps WP10a and WP4 parallel.
- **Cost of changing later:** low; renaming keys in about 60 chain items. Three findings (register double buffering, FlashAttention-2 phase-offset slot, runtime num_groups) depend on these assumed variables.

## D105 — Three new Patterns proposed

- **Question:** D50 says new Patterns need your confirmation; the decompositions needed transactions the 8 seed Patterns cannot describe.
- **Choice:** proposed `work_assign` (assignment and order of work units), `issue_order` (issue order within / across executors) and `scan` (prefix computation), all with category `other` because the schema's category enum has no matching entry; they stay `proposed`.
- **Rationale:** 7 chain steps over 5 references needed work_assign, 2 needed issue_order (plus one unmatched), 1 needed scan.
- **Cost of changing later:** low while proposed; if the category enum gains `schedule` / `scan`, the three files change one field.
