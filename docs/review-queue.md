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

## D181 — Worker deployment and commit match

- **Question:** conventions §5 require worker and control plane at the same commit; how is that enforced without a shared filesystem?
- **Choice:** each commit is deployed to `<root>/src/<commit>` with its own venv; the ssh transport only runs the deployment of the control plane's commit; registration and the Executor refuse other commits.
- **Rationale:** a worker can never run code other than the control plane's; old deployments stay for reproducing earlier records.
- **Cost of changing later:** low: deployment layout is in `scripts/worker-setup` and `SshTransport.script`; disk use grows with deployments (cleanup not implemented).

## D184 — One in-process Executor per control-plane process

- **Question:** where do the job queue, pool limits and GPU leases live?
- **Choice:** in the `JobExecutor` object of the process that submits; the GPU lease is mirrored in the ops `locks` table for cross-process exclusivity; gpu-shared deferral is per process.
- **Rationale:** simplest design that meets the WP2 acceptance; the orchestrator (WP6) can own a single executor for the whole control plane.
- **Cost of changing later:** medium: a shared queue (ops table or API service) would move `_place`/`_acquire_slot` onto a table; the Executor protocol stays.

## D186 — Uniform harness through the CUDA driver API

- **Question:** "all backends' modules are loaded and launched the same way" — as `.so` or cubin, and from what host code?
- **Choice:** cubins only, launched by `kta.backends.harness.run` via ctypes on libcuda (no torch, no compiled launcher); timing with CUDA events per launch.
- **Rationale:** both backends produce cubins naturally; one launch path makes timing comparable; no extra dependencies on workers.
- **Cost of changing later:** low for adding `.so` modules; WP3 may refine the timing protocol (events per launch include launch overhead).

## D187 — TileLang in a container, compiled by the host nvcc

- **Question:** how to install and pin TileLang on an aarch64 GB10 worker without system installs, and keep one toolchain for both backends?
- **Choice:** a worker image from the NGC PyTorch 26.03 image (pinned digest) plus a pinned pip lock; codegen in the container without network or GPU; compile with the host CUDA 13.0 nvcc mounted read-only.
- **Rationale:** TileLang needs torch/TVM wheels that are heavy for the project venv and would not resolve on the macOS control plane; the container's own CUDA 13.2 needs forward-compatibility mode, so cubins built by it may not match the host driver.
- **Cost of changing later:** low: `docker/worker/`, `kta.backends.tilelang`; changing TileLang version means regenerating `requirements.lock` (`docker/lock`).
- **Status:** the user accepted the container approach on 2026-10-30 (via the coordinator); the pinned base, TileLang-only content and host-CUDA mount remain defaults.

## D188 — What `support: yes/no/unknown` means for a HwFeature

- **Question:** the spec says "compile and run a minimal kernel: success -> yes, compile failure -> no"; what about features that cannot be exercised without host setup, or runs that fail?
- **Choice:** compile-only entries and failed runs are `unknown`; the probe uses `-arch=sm_121a`; raw output is kept per feature.
- **Rationale:** never claims support that was not executed; `no` only when the toolchain rejects the instruction for sm_121a.
- **Cost of changing later:** low: one function (`_decide`); adding run harnesses for TMA tensor maps would turn `unknown` into `yes`/`no`.

## D198 — What the WP2 gate checks

- **Question:** spec/gate.md asks for `pytest -m gpu tests/acceptance/WP2/` via the Executor on spark and real Target/Build records.
- **Choice:** the gate runs that suite (it reaches spark and slai over ssh, so it needs this commit deployed on both) and checks listed, real, clean Target, Build, Worker, EnvManifest and Check records in `$KTA_DATA/dev/WP2/kta.db`, including a passed vector-add Check per backend.
- **Rationale:** the acceptance run proves the behaviour now; the record check proves the delivered evidence.
- **Cost of changing later:** low: one function in `tools/kta_gate/specific.py`.
