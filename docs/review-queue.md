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

## D164 — Format of `src/kta/orchestrator/machines.yaml`

- **Question:** D96 makes one YAML file the definition of the solve loop, expand loop and task lifecycle for both the frontend and WP6's `step`, but the design documents give tables, not a data format.
- **Choice:** states with `kind`/`work`/`does`; transitions with `kind`, `rule`, `creates`, `when`, short `label`; `from: "*"` for "any state"; escalation after the round limit is an instance flag, not a state; unspecified paths are written under `notes` instead of guessed. Two such notes are open: (1) what happens when reproduction misses ε in the expand loop; (2) after a task created at `judging` (T6, T4, T1/T2) is accepted, design/10 §2.1 says "resume at the creating state" — whether building and measuring rerun first is not said.
- **Rationale:** keeps the diagrams and the orchestrator on one definition and makes gaps in the design visible on the Overview page.
- **Cost of changing later:** low before WP6 implements `step`; afterwards the consistency test and the frontend change together.

## D165 — Frontend information architecture and Overview layout

- **Question:** you asked for a readable frontend whose home page shows the state-machine flowcharts and current work; the exact page split and layout were left to the assistant (D93).
- **Choice:** see web/DESIGN.md: Overview = solve loop (full width) → Now band (agent slots, GPU lease, pools, workers, budget, inbox) → expand loop → task lifecycle → pipeline strip; other pages each have one primary visual (resource board, swimlanes, iteration timeline, evidence chain, chain DAG, discovery curve).
- **Rationale:** the solve loop needs the full width to stay legible (9 layers); the Now band answers "what is happening" right under it.
- **Cost of changing later:** low; layout lives in page components only.
