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
