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

## D205 — Pattern parameters (`p.*`) in Template expressions and chain bindings

- **Question:** a Template's capacity and lifetime conditions need the size and timing of the transaction it refines (bytes of a tile, capacity of its level), which are neither Template variables nor workload parameters. Chain bindings in WP10a also hold strings that are descriptions or expressions, not family variables.
- **Choice:** Template expressions refer to the refined transaction's parameters as `p.<param>`; a chain item may bind `p.<name>`; constant bindings are checked against domains and C/Pre, string bindings only in families (must name a family variable or domain value).
- **Rationale:** keeps the five paper steps self-contained and lets WP10a's reconstruction chains validate without rewriting them.
- **Cost of changing later:** medium: WP5 must decide how a family binds `p.*` to expressions over family variables; changing the convention touches the seed Templates and every chain that binds `p.*`.

## D206 — Content of the five seed Templates

- **Question:** the paper gives the five GEMM steps in a table and one full listing (shared-memory tiling); fields such as levels, change kind, cost terms, Γ and constraint forms had to be chosen.
- **Choice:** variable names from WP10a's assumed seed (D103); levels 0→1, 1→2, 2→3, 2→3, 4→5; `async_transfer` is a representation expansion; capacity/lifetime constraints over `p.*`; step 5's chip variables fixed through `hw.*`; obligations listed with planned methods.
- **Rationale:** follows the paper's tables and listing; compatible with WP10a's seven chains; the expression language cannot compare strings (D216).
- **Cost of changing later:** low before WP7 builds the GEMM family; afterwards every chain using the seed needs a new Template version.

## D213 — Family generators run in the control-plane process during validation

- **Question:** `validate_family` must check code markers in generated source and that `generate()` returns for edge shapes; the generator is worker-written Python.
- **Choice:** load `gen.py` with importlib in the calling process; only source text is produced; compiling and running go through the Executor.
- **Rationale:** generation is pure text production and fast; an Executor round trip per validation would need WP2.
- **Cost of changing later:** low: one function (`kta.knowledge.families._generation`) can be switched to an Executor job once WP2 is merged.

## D215 — What the WP4 gate checks

- **Question:** spec/gate.md asks for "seed 全部通过校验". After WP10a merges, `knowledge/seed/references/` is part of the seed tree, but its Templates fail validation for reasons outside WP4 (string comparisons, HwFeatures not yet in the store, a Problem name defined twice, unmet `requires` relations).
- **Choice:** FAIL when the WP4 seed (Sources, Patterns, Structures, the five Templates) does not import and validate or when the named acceptance tests did not pass; the references are imported in the same scratch store and reported as WARN with the issue counts.
- **Rationale:** WP4's acceptance is about its own seed; the reference findings are listed in deliveries/WP4.md "Needed changes" for WP10a, WP2 and core.
- **Cost of changing later:** low: switch the WARN to FAIL in `tools/kta_gate/specific.py` once those changes land.

## D216 — No string values in expressions

- **Question:** `kta.core.expr` accepts only numbers and booleans as literals, so `mechanism == "instruction"`-style conditions cannot be written; WP10a's Templates use them (13 expressions fail to parse).
- **Choice:** the seed Templates avoid string comparisons and state such conditions in obligations and Γ text; a change to core is requested (string literals in `==`/`!=`, optionally `in` over a literal list; not linear).
- **Rationale:** `src/kta/core` is owned by WP1 and locked for WP4.
- **Cost of changing later:** low for core (one node type in the checker/evaluator); after the change the seed Templates can gain conditional constraints in a new version.
