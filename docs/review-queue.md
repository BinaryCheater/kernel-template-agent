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

## D169 — Record conventions the API reads

- **Question:** the pages (case timeline, evidence chain, running jobs, GPU lease, coverage) must be derived from records and ops tables, but the schemas do not say which refs, Report titles, span attrs or loop data keys connect them.
- **Choice:** the table in spec/modules/api-web.md "Record conventions" (e.g. Solution refs case/family/measurement/check/bound; judgement Report with data.units and explained_share; loop_step spans per state; job spans with job_id/pool/worker; lock `gpu:<worker>:<device>`; Worker label `pools`).
- **Rationale:** without one written convention each producer would invent its own and the frontend could not show their work; the fixture world (D168) exercises every convention.
- **Cost of changing later:** low until WP2/WP6/WP7 write records; afterwards a migration of the views (records are immutable).
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

## D223 — Format of `Problem.reference_impl`

- **Question:** the numeric check needs an FP64 reference computed on the GPU; the schema only says `reference_impl: str`.
- **Choice:** a Python statement over torch tensors named like the Problem's tensors, with the shape variables in scope (`C = A @ B`), executed in the worker image (PyTorch, no network) on the GPU in FP64 from the exact kernel input bytes. `none: <reason>` marks a timing-only problem.
- **Rationale:** works for any operator torch can express, needs no per-problem reference kernel, and runs where the GPU is; the image already exists for TileLang.
- **Cost of changing later:** low: `kta/checks/reference.py` and its script; Problems written with this format would need rewriting.

## D224 — measure and checks return the record ID with the body

- **Question:** the spec signatures return `Measurement` / `Check` bodies, but callers (orchestrator, reports) need the record IDs, and `Check` has no `error_kind` field for checker errors.
- **Choice:** `Measured(id, body, times_us)`, `Comparison(..., report_id)`, `Checked(id, body)`; a checker error is a failure item with `error_kind: "checker_error"`, exposed as `Checked.checker_error`.
- **Rationale:** content-hash IDs cannot be recovered from a body without the envelope; adding a schema field would touch WP1's schemas.
- **Cost of changing later:** low until WP6 consumes these types.

## D226 — Measurement protocol versions and who writes them

- **Question:** what `measure-v1` and `measure-v2` contain, and who "approves" a Protocol (the schema says human-approved).
- **Choice:** v1 = the spec table (+ a 100000-iteration cap per phase, no clock locking since it needs root, noise floor 1.1% provisional from WP13 finding 6); v2 = v1 with the noise floor measured by the WP3 noise probe (largest CV of 30 group medians across the probed kernels, rounded up to 0.1%). Both are written by the tool actor; this review is the human approval.
- **Rationale:** follows the spec's "revise after measuring noise, new version" rule; everything else is unchanged so v1 and v2 results differ only in the significance threshold.
- **Cost of changing later:** low: a new version (`measure-v3`) and its records; existing Measurements keep naming their version.

## D227 — full_check needs `build_for` and `case_id`

- **Question:** a module is compiled for one shape (fixed launch grid), but full_check must also run edge and hidden shapes, and check the Contract identity against the Case record; checks may not import knowledge.
- **Choice:** keyword-only `case_id`, `shape` and `build_for(shape) -> (GeneratedKernel, BuiltModule)`; without `build_for` full_check raises NotSupported. `SpaceItem` gains the same fields.
- **Rationale:** generation and compilation stay with the orchestrator (as spec'd for verify_space); the checker stays independent of knowledge.
- **Cost of changing later:** moderate once WP6 calls it; a shape-generic launch (grid as an expression) would make `build_for` optional.

## D228 — No Nsight Compute counters on spark

- **Question:** `profile` needs GPU performance counters; spark's NVIDIA driver has `RmProfilingAdminOnly=1`, so ncu fails with ERR_NVGPUCTRPERM for the worker user.
- **Choice:** `profile` is implemented (ncu job, CSV parse, Profile record) and raises NotSupported when ncu reports missing permission. Nothing on spark was changed.
- **Rationale:** enabling counters means changing a driver parameter on spark (system configuration) or running privileged containers; both are the user's call per AGENTS.md.
- **Cost of changing later:** none in code: once counters are allowed, `profile` writes Profile records.

## D232 — What the WP3 gate checks

- **Question:** spec/gate.md asks for `pytest -m gpu tests/acceptance/WP3/` via the Executor on spark and real Measurement, Check and Protocol records.
- **Choice:** the gate runs that suite (four named tests; needs this commit deployed on spark) and checks listed, real, clean records in `$KTA_DATA/dev/WP3/kta.db`: Measurement and Check from a GPU worker, measure-v1 and v2, v2's noise reports (≥ 30 groups), the checker-selftest Report with all four variants as expected, and a significant and a non-significant compare under v2.
- **Rationale:** the acceptance run proves the behaviour now; the record check proves the delivered evidence.
- **Cost of changing later:** low: `tools/kta_gate/specific_wp3.py`.

## D241 — What a Prediction holds

- **Question:** the Prediction schema has `per_unit`, `serial`, `total`, `model` without units or a statement of whether efficiencies are applied.
- **Choice:** `per_unit` and `serial` are times at peak (eff = 1, no occupancy factor) in µs; `total` is the model's prediction in µs.
- **Rationale:** any model version can be refitted on predictions made by another; the bottleneck unit at peak stays visible.
- **Cost of changing later:** low before WP7 stores real predictions; afterwards a migration or a schema version.

## D243 — Minimum work for the bound

- **Question:** `bound()` needs the semantic minimum work, computed "from the Problem's reference definition"; the Problem schema has no field for it.
- **Choice:** `Problem.params["min_work"]` maps unit -> expression over `case.*`; peaks from `HwModel.peaks`.
- **Rationale:** no schema change; the Problem author states it once per problem.
- **Cost of changing later:** low: `kta.models.bound.min_work`.

## D244 — Fitting rule and interaction significance

- **Question:** the spec asks for interaction terms "when the combined residual is significant by measure's rule", but models (layer 4) cannot import measure (layer 5), and measure is not merged.
- **Choice:** own F test (F > 4, >= 3 residual dof, >= 10% lower loss); relative-error loss; every fifth point held out.
- **Rationale:** a standard nested-model test with a conservative threshold; hold-out error is reported per version as the spec requires.
- **Cost of changing later:** low: `kta.models.fit`; or move the rule into core so both modules share it.

## D245 — Model versions as Report records

- **Question:** "each model version is a record", but there is no UnitModel kind.
- **Choice:** a `Report` with subject `unit_model`; refs `previous`, `predictions`, `measurements`.
- **Rationale:** no new kind in core; Reports are already queried by subject.
- **Cost of changing later:** medium once WP6/WP9a read them: a new kind needs a migration of existing versions.

## D246 / D247 — Binding semantics and the cost scale `p.instances`

- **Question:** WP4 left the meaning of `p.*` bindings in families to WP5 (D205); Template cost terms also lack a statement of scale (per instance or per kernel).
- **Choice:** string bindings are family variables, list-domain values, or expressions over family variables, `case.*` and `hw.*`; `p.*` likewise; the reserved `p.instances` multiplies the item's cost terms (Template costs describe one instance).
- **Rationale:** lets a family state per-tile sizes (alignment, capacity) and whole-kernel counts without changing the Templates; the GEMM test family over the five seed Templates needs both.
- **Cost of changing later:** medium once WP7/WP10b families use them. `validate_family` (WP4) accepts all of this already, except that it reports an expression bound to a *Template variable* (not a `p.*`) as `unbound_name`.

## D250 — Rerank inputs

- **Question:** `rerank(candidates, builds, ...)` keys builds by a string and needs threads per CTA for occupancy, which Build does not record.
- **Choice:** key = `Candidate.key` (hash of bucket and assignment); threads from the family variable named `threads`.
- **Rationale:** stable without store IDs; families that tile CTAs name their thread count anyway.
- **Cost of changing later:** low: `kta.solve.select`.

## D252 — What the WP5 gate checks

- **Question:** spec/gate.md asks for "seed validates" and `pytest tests/acceptance/WP5/`; the seed has no family yet.
- **Choice:** the named acceptance tests; WP4's seed check; every seed family plus the WP5 GEMM test family validates and assembles, and together they use all five seed Templates.
- **Rationale:** checks that the seed's Templates compose into a formulation, not only that each validates.
- **Cost of changing later:** low: one function in `tools/kta_gate/specific.py`.

## D261 — Where a waiting solve loop resumes

- **Question:** docs/design/10 §2.1 says a waiting loop "returns to the state that created the task". For tasks created in `judging` (T6, T4, T1/T2) that would judge the same measurements again.
- **Choice:** T1, T2 and T6 resume at `retrieving` (the pool changed); T4 resumes at `solving` (only the model changed). Each resume is a new round; at the round limit the loop stays in `waiting`, escalated. `machines.yaml` drops `waiting -> judging` and gains `waiting -> solving`.
- **Rationale:** every round then rebuilds, rechecks and remeasures with the new knowledge or model.
- **Cost of changing later:** low: `RESUME_AT` in `src/kta/orchestrator/solve_loop.py` and two YAML edges.

## D262 — Solve loop edges for empty results

- **Question:** the design has no path for "no feasible candidate" or "nothing built".
- **Choice:** `solving -> waiting` creating T1 (extend the family) and `building -> waiting` creating T6 (fix the family).
- **Rationale:** both mean the family, not the model, is wrong; going on to judging would have nothing to judge.
- **Cost of changing later:** low (two edges and two branches).

## D263 — Judging rules, concretely

- **Question:** rules 2 and 3 of docs/design/10 §2.1 have no numbers ("severely out of order", "gap explained").
- **Choice:** rule 2: Spearman rho < 0.5 with at least 3 measured, or median relative error > 0.3. Rule 3: not slower than the strongest baseline beyond 3 x the protocol noise floor, or the model explains >= 80% of the gap to the bound. Rule 4: T2 when a Reference of the Problem exists, else T1. One family per loop instance (the best retrieved). All thresholds are config `[orchestrator]`.
- **Rationale:** first-fill values, recorded in every judgement Report, so later tuning can be traced.
- **Cost of changing later:** low for the numbers (config); medium for the definitions (one function, `judge`).

## D265 — Task lifecycle storage in the ops table

- **Question:** WP1's ops `tasks` table knows `queued`, `leased`, `done`, `failed`, `cancelled`; the lifecycle (10 §3.2) has eleven states.
- **Choice:** the column now holds the lifecycle names directly, changed by compare-and-set (`SqliteOps.update_task`, additive). The orchestrator never writes `done`/`failed`. WP9a's `TASK_STATE` map passes unknown names through, so its pages show the real states; its `escalated` heuristic becomes unnecessary.
- **Rationale:** one source of truth for the lifecycle the frontend draws.
- **Cost of changing later:** low; WP9a may drop its mapping.

## D266 — Task lifecycle additions

- **Question:** 10 §7 says a crashed checker suspends the task without counting a retry, and 11 §6 says lost leases return to the queue; neither is in the lifecycle diagram.
- **Choice:** new state `suspended` (`checking -> suspended`, `suspended -> checking` by you) and edge `leased -> queued`.
- **Rationale:** both paths exist in the design text; now they are visible in the diagram.
- **Cost of changing later:** low.

## D267 — Expand loop open paths

- **Question:** 10 §2.2 does not say what happens when reproduction misses epsilon.
- **Choice:** write a Gap (missing family) and create T1; its acceptance resumes at `reconstructed`. After extraction tasks a fresh T2 re-matches the chain.
- **Rationale:** a missed reproduction means the family lacks a step the reference has.
- **Cost of changing later:** low; nothing exercises the path until WP11 builds references.

## D274 — CLI command set

- **Question:** docs/design/10 §5 names `kta solve <template> <case>` (agent) and spec lists `kta solve <case>` (start the loop); WP3 already owns `kta measure` / `kta check` groups.
- **Choice:** `kta solve <case>` starts the loop. Agent compile/run commands (`instantiate`, `build`, `sass`, `run`, `src`) raise NotSupported until WP8; `bound` is absent. Notes are Reports titled `note`.
- **Rationale:** agents do not exist yet and the loops compile, check and measure candidates themselves.
- **Cost of changing later:** medium for WP8, which defines the agent workspace and the remaining commands.

## D275 — What the WP6 gate checks

- **Question:** spec/gate.md asks that "every decision rule of the transition function has a test".
- **Choice:** named tests (step produces exactly the YAML edges; created task types are on the edge; every lifecycle edge occurs; refused reservation keeps the task queued) plus one passing `test_judging_rule_<n>_*` per rule number found in machines.yaml.
- **Rationale:** a new rule or edge in the YAML fails the gate until it is tested.
- **Cost of changing later:** low: one function in `tools/kta_gate/specific.py`.
## D300 / D301 — Where the agent CLI runs, and how its `kta` reaches the store

- **Question:** spec/modules/agents.md puts the agent process on the control-plane machine, while the WP8 acceptance names spark's user-level Codex directory and DGX OS GPU permissions, and conventions §6 sends all computation through the Executor. With the control plane on the Mac and the agent on spark, the agent's `kta` calls have no direct path to the store.
- **Choice:** the CLI runs as an Executor job (pool `agent`) on the worker named by `[agents] worker` (spark), started by `kta.agents.launch`; workspace and CLI home live in that host's data directory. The agent-side `kta` authenticates with `KTA_TASK_TOKEN` and talks to the control-plane API at `KTA_API`; the API verifies the token with `kta.agents.guard.verify_task_token`. Setting `[agents] worker` to a `local` worker runs agents on the control host instead, where `kta` can open the store directly.
- **Rationale:** keeps the agent's shell commands (and fan noise) off the Mac, matches the acceptance items, and gives every agent process the Executor's timeout, process-group kill and output transfer by hash.
- **Cost of changing later:** low for the agents module (one config value); the CLI client mode (WP6) and the API token endpoint (WP9) are needed in either case once the kta-agent user cannot read the store.

## D305 — Task token

- **Question:** the spec requires a task token in the agent's environment but defines neither its form nor where it is checked.
- **Choice:** HMAC-signed `kta1.<task>.<expiry>.<mac>`; key file `~/.config/kta/task-token.key` (600) created on first use on the control host; valid until 3× the wall budget and only while the task is leased or running.
- **Rationale:** no new table or migration; revocation follows the task state.
- **Cost of changing later:** low: `kta.agents.guard`; tokens live only as long as a task.

## D307 — Codex sandbox with network access

- **Question:** the agent-side `kta` needs the network (D301); Codex's `workspace-write` sandbox blocks it by default.
- **Choice:** `workspace-write` with `network_access = true`, `approval_policy = "never"`; no firewall in the first phase; direct network tools (curl, wget, ssh) are counted as violations.
- **Rationale:** the only setting under which `kta` works inside the sandbox without giving up the write restriction.
- **Cost of changing later:** low (one line in the profile's config.toml, a new profile version); restricting egress to the API and the model provider needs a host firewall rule (system change).

## D308 — Model of codex-v1

- **Question:** which model and reasoning effort the first profile uses (D02 is still open).
- **Choice:** `gpt-5.4`, effort `medium`, CLI 0.117.0: what spark's Codex is configured with today.
- **Rationale:** a model the installed CLI is known to serve; changing it is a new profile version.
- **Cost of changing later:** low (new profile version, pointer switch by you).

## D310 — Credentials live on the agent host

- **Question:** conventions §8 puts subscription credentials in a protected directory on the control-plane machine; under D300 the CLI runs on spark.
- **Choice:** `[agents.hosts.spark] auth_dir = ~/.config/kta/auth/codex` on spark, symlinked into each CLI home, never archived. Placing the credential file there is your step (copying `~/.codex/auth.json` on spark, or `codex login` with that CODEX_HOME).
- **Rationale:** the CLI must read them where it runs; nothing copies credentials between machines.
- **Cost of changing later:** low.

## D312 — kta-agent user and permissions

- **Question:** D09's default isolation `config+user` needs a system user, a sudoers rule and ACLs on spark.
- **Choice:** `scripts/kta-agent-setup` (dry run by default, `--apply` changes the system): group `kta-agents`, system user `kta-agent` without login, setgid group directories for workspaces and CLI homes, traverse-only ACLs, `/etc/sudoers.d/kta-agent`; GPU device modes are reported and left unchanged — if they are world-accessible, the launcher's GPU guard (detect and terminate) enforces "GPU only via kta".
- **Rationale:** smallest change that separates the agent from the store and the user's files; device modes are system-wide driver settings.
- **Cost of changing later:** medium: removing the user and ACLs is a second system change.

## D317 — Codex's sandbox on spark

- **Question:** Codex 0.117's default Linux sandbox (bubblewrap) fails on spark because Ubuntu 24.04 restricts unprivileged user namespaces (`kernel.apparmor_restrict_unprivileged_userns = 1`); every sandboxed command would fail.
- **Choice:** codex-v1 enables `features.use_legacy_landlock`. Probed with `codex sandbox linux` (no model call): writes inside the workspace succeed, writes outside are denied, network is blocked by default; reads are not restricted.
- **Rationale:** needs no system change; the read restriction comes from the kta-agent user (D312) in either case.
- **Cost of changing later:** low (one config line, new profile version). The alternative — an AppArmor profile or sysctl allowing bwrap's user namespaces — is a system setting you would have to approve.

## D322 — How batch agent environments are generated

- **Question**: what makes a batch's agent workspace "completely generable", and where does it live?
- **Choice**: one AgentProfile id plus a code commit determine the batch. Skills are deterministic tar blobs and templates are blobs. `kta agents env build <batch>` writes `$KTA_DATA/<env>/batches/<batch>/` with `env.lock.json`; `verify` regenerates and compares before every run; a batch is never overwritten; credentials are linked per work unit and are never stored.
- **Why**: the profile is already the single record of agent behaviour (agents.md "Controlled configuration"). Storing content instead of paths survives source changes, and regenerating plus comparing makes isolation checkable instead of assumed.
- **Cost of changing later**: low until WP12. The layout and command names live in the agents module and the CLI only. Batches already built keep their lock files.

## D280 — The GEMM contraction and operand traffic are costed by two seed Templates (WP7)

- **Question:** WP5 found that the seed GEMM chain costs neither the contraction nor the operand DRAM/L2 traffic, so the top candidates tie. Fix it with Templates or with a family-level cost term?
- **Choice:** Templates. WP10a's proposed `output_tiling@1` and `compute_instruction_selection@1` enter the seed as version 2 with cost blocks: launch slots, L2 bytes per tile and compulsory DRAM bytes; tensor FLOPs and instruction issue per executor tile. Γ of the compute step compares the operand type with the instruction's through numeric `p.*` params, so `reuse_plan` flags a BF16 case. The `contract` and `map` Patterns gain the matching params.
- **Rationale:** a family-level cost needs a schema change (Family has no cost field), and the costs belong to steps every GEMM-like chain has (24 §3.1 lists both steps).
- **Cost of changing later:** medium: the seed has seven Templates now; WP4/WP5 unit tests count them; WP10a's references keep version 1, and stores created before this change hold the old `contract`/`map` bodies (re-import conflicts there).

## D283 — Per-unit peaks from six microbenchmarks; `issue` from the FMA run; `smem_queue_depth = 1` (WP7)

- **Question:** the solve needs an HwModel with measured peaks; nothing wrote one, and the Templates' units mix counts (thread-level copy operations, warp-level instructions, barriers).
- **Choice:** one CUDA microbenchmark per unit under the measurement Protocol; issue = warp instructions/s of the FFMA run; model assumptions `smem_queue_depth = 1`, 32 banks of 4 bytes.
- **Rationale:** real numbers with the same timing path as candidates; the assumptions are the values the seed's `shared_resource_binding@1` needs and no device query gives.
- **Cost of changing later:** low: a new HwModel version; predictions name the HwModel they used. The issue unit's mixed counting (async_transfer counts thread-level copies) is a seed-modeling question for the method discussion.

## D284 — torch.matmul baseline: protocol replica in the worker image (WP7)

- **Question:** WP3's bench job loads cubins and host libraries only; torch.matmul cannot run there.
- **Choice:** a self-contained script in the PyTorch worker image that repeats the bench loop step for step; the baseline is a Report fixed before measuring and named by the Case.
- **Rationale:** same inputs, flush, NaN fill, event timing and stopping rule as candidates, with the library the spec names.
- **Cost of changing later:** low: `kta.measure.baseline`; a cuBLAS-only baseline (`.so` calling cublasGemmEx) would use the existing bench job but is not what the spec asks for.

## D285 — compute-sanitizer runs on a small shape (WP7)

- **Question:** `full_check` runs memcheck and racecheck at the module's own shape; racecheck on a 4096³ GEMM takes far too long, and docs/design/50 §3 says sanitizers run on small shapes.
- **Choice:** optional `sanitizer_shape`; WP7 passes 200×136×72 (not a tile multiple, so residue tiles are exercised).
- **Rationale:** follows the design text; the default keeps WP3's behaviour.
- **Cost of changing later:** low; WP6's `work_checking` should pass it too.

## D287 — Case.target is the Target name (WP7)

- **Question:** WP3's selftest Cases use the Target record ID; HwModel.target and the unit-model Reports use the name, and WP6 looks HwModel up by `Case.target`.
- **Choice:** WP7 Cases use the name `gb10`.
- **Rationale:** matches HwModel and model lookups without a join; `target_sm` accepts both.
- **Cost of changing later:** low, but WP6 and every Case writer must agree.

## D288 — What the WP7 gate checks (WP7)

- **Question:** spec/gate.md asks for a complete solve-loop record, three failed fault-injection Checks and a consistent replay.
- **Choice:** static checks (unit tests, family, Case file) plus listed, real, clean records: judgement or Solution, `wp7_faults` with all three caught (failed Checks from a GPU worker), replay Report consistent, HwModel, baseline Measurement, `wp7_reuse` as expected. Code stage: WARN.
- **Rationale:** each acceptance item of WP7 maps to a record the gate can find.
- **Cost of changing later:** low: `tools/kta_gate/specific_wp7.py`; the judgement/replay record shapes follow WP6 and may need adjusting after it merges.
## D342 — How a HumanAction carries its parameters

- **Question:** HumanAction has only `action` and `reason`; a budget change or a priority needs its values in the record, and refs can only hold record IDs.
- **Choice:** `action = "<verb> <compact sorted JSON>"`, the subject in refs, the actor `human` named by `[api] operator` (default `user`). The log endpoint parses it back.
- **Rationale:** no schema change for a code-stage WP; the record stays self-contained. WP6 already writes `task cancel <id>`, so two forms coexist until one is chosen.
- **Cost of changing later:** low now: add `params: dict` to HumanAction (schema version 2) and write both forms during a migration; the frontend reads the parsed form only.

## D343 — Flagging a record sets the status `suspect`

- **Question:** design/30 §3 says "flag as suspect: triggers re-verification, does not change the status", while the schema has a human-only status `suspect`.
- **Choice:** write a StatusChange to `suspect`; for a Solution or a Measurement, Check or Bound a Solution rests on, ask WP6 to re-verify it (a new solve loop per Case, WP6 D368). Other kinds are flagged only. (Revised in WP9b round 3.)
- **Rationale:** the status is visible everywhere the record appears; without it nothing would ever set `suspect`.
- **Cost of changing later:** low: drop the StatusChange in `kta.api.operate.flag`; existing suspect statuses can be reset by a human StatusChange.

## D345 — Record conventions for Experiments, Improvement and Pool pages

- **Question:** the pages need to find experiment arms, results, improvement items and confirmations in records that WP10b, WP11 and WP12 have not written yet.
- **Choice:** see D345 in docs/decisions.md (Protocol `type: experiment` with `params.arms`, ExperimentRun `condition`, Report titles `experiment` and `improvement`, HumanAction `approve_protocol`, `confirm_pattern`).
- **Rationale:** uses existing kinds and fields only; mirrors D169 for the read-only pages.
- **Cost of changing later:** low on the API side (`src/kta/api/studies.py`); producers must follow whatever is chosen.

## D346 — What the full WP9b gate requires

- **Question:** the workplan's acceptance "start a T1, cancel a task, reject a result from the frontend, records correct" must become a gate check.
- **Choice:** listed, real, clean human HumanActions in `dev/WP9b` with action prefixes `create_task`/`task create`, `cancel`/`task cancel`, `reject`, plus a screenshot path in the delivery note.
- **Rationale:** the HumanAction is the record every operation writes; prefixes cover both the API's and WP6's current forms.
- **Cost of changing later:** low: one function in `tools/kta_gate/specific_wp9b.py`.

## D348 — WP9b pages and navigation

- **Question:** where the intervene pages and the operation controls go.
- **Choice:** sidebar Watch / Understand (adds Pool, Model) / Intervene (Inbox, Operations, Experiments, Improvement, Agents); controls appear next to their subject (record drawer, Running, Task, Result), and the Operations page collects start-task, pause, budgets, the action log and the capability table.
- **Rationale:** D93 (readable, one focus per page); every control says what it writes, and unwired ones say what they need.
- **Cost of changing later:** low: `web/src/components/Shell.tsx` and the page files.

## D360 — Which small shape compute-sanitizer runs on in the solve loop

- **Question:** WP7 (D285) passes a fixed 200×136×72 for GEMM; the loop needs a rule for any Problem.
- **Choice:** the Contract's first edge shape with every dimension above 1 (129×193×65 for the seed GEMM cases).
- **Rationale:** edge shapes already exist per Contract, include a non-tile-multiple one, and are small.
- **Cost of changing later:** low: `pipeline.sanitizer_shape`.

## D364 — Pause semantics

- **Question:** WP9b's pause/resume of global, process, loop and task scopes needed a meaning.
- **Choice:** a paused scope stops new work below it (process ticks, loop steps, task leases); running work finishes, as for an exhausted quota (11 §3.4).
- **Rationale:** no work is cut off mid-job, and resuming is just clearing the mark.
- **Cost of changing later:** low: `kta.orchestrator.control` and three call sites.

## D368 — How a suspect record is re-verified

- **Question:** the inbox says "re-verify" for suspicious results, but nothing defined what runs.
- **Choice:** a fresh solve-loop instance for the Case of the suspect Solution (or of the Solution resting on a suspect Measurement/Check/Bound). The maintain process starts it once per suspect record. Other kinds are marked unsupported.
- **Rationale:** the loop already rebuilds, rechecks and remeasures with new records; no second verification path.
- **Cost of changing later:** medium: a narrower re-check (only the suspect Check or Measurement) needs a per-kind job.

## D370 — CLI client mode inside an agent workspace

- **Question:** D301 says the agent's `kta` talks to the API with its token, but the API has only `POST /write/record`.
- **Choice:** with `KTA_TASK_TOKEN` set, only `kta note` works (through that endpoint). Every other command refuses with NotSupported naming the endpoint it needs, and no command opens a local store.
- **Rationale:** an agent must not fall back to direct store access; missing endpoints are visible.
- **Cost of changing later:** low on the CLI side (`kta.cli.client.dispatch`). The API (WP9b) needs read, query, validate, build and submit endpoints for agents to work.
- **Update (D379):** WP9b added those endpoints (`wp/WP9b` @ c890af8); client mode now uses all of them, and only `run` and `src` refuse.
## D382 — Expand-loop tasks follow the chain, not the Gaps

- **Question:** spec/modules/orchestrator-cli.md says each Gap of a T2 creates one T3. WP10a's 55 Gaps map to 30 Templates (several Gaps per proposed Template), and 18 Gaps (compiler-decided allocation, family-level dispatch, a missing check kind) have no Template to extract.
- **Choice:** one T3 per Template the accepted chain names that the pool lacks; Gaps are recorded and counted in coverage but create no task; those not tied to a chain Template stay open. T1 is created only after a missed reproduction.
- **Rationale:** a machine-readable link from a Gap to "its" Template does not exist (only Gap text says "Proposed: X@1"); per-Gap T3s would produce duplicate Templates or tasks with nothing to deliver.
- **Cost of changing later:** low in code (`expand_loop._analyzed`, `expand_work.work_waiting`); a per-Gap rule needs a Gap → chain item link in the schema or the T2 layout.

## D384 — A cancelled T3 leaves its Template open

- **Question:** how the `gap` state ends when an extraction cannot be done.
- **Choice:** a T3 you cancel counts as settled; if any T3 of the round was accepted a new T2 re-matches (bounded by `max_rounds`), otherwise the loop continues to `reconstructed` with the Gaps open.
- **Rationale:** agents cannot decide to skip extraction (only you can cancel); without this the loop escalates forever on a Template nobody will write.
- **Cost of changing later:** low: `expand_loop.step` and one machines.yaml edge.

## D385 — Analysis-only expand loops

- **Question:** with real runs paused (D320), runnable references cannot be built or measured, so their loops block in `frozen`.
- **Choice:** `kta expand --analysis-only REASON` (or an empty `runnable_on`) skips build, measurement, reproduction and ablation; the loop reaches `covered` without a reference point or reproduction error.
- **Rationale:** docs/design/24 §2 already allows "analysis only" for references that cannot run on GB10; this extends it to a recorded human choice.
- **Cost of changing later:** low; but a `covered` reached this way is not a reproduced reference, and WP10b's acceptance ("frozen to covered") should say whether analysis-only counts.

## D387 — `covered` takes a pool snapshot and writes the coverage page

- **Question:** the coverage page (WP9b) reads a Report keyed by a PoolSnapshot, but nothing creates snapshots yet (the learn process is WP11).
- **Choice:** the expand loop snapshots all knowledge records when it reaches `covered`, moves `current_pool` by compare-and-set, and writes the coverage Report in the page's shape.
- **Rationale:** makes the coverage page show real data now (workplan WP10b), with no API change.
- **Cost of changing later:** medium: if WP11 owns snapshotting, `expand_work.write_coverage` should call its function; the record shapes stay.
## D400 — Pool snapshots and the learning backlog

- **Question:** what a PoolSnapshot contains and when the learn process writes one.
- **Choice:** all usable knowledge records (Pattern, Structure, Template, Chain, Family, Reference) after any accepted task with outputs; backlog = ops pointer filled by `kta learn add`.
- **Rationale:** snapshots then equal what retrieval sees; no separate admission rule exists yet.
- **Cost of changing later:** low: `members()` and `needs_snapshot()` in `src/kta/orchestrator/proc_learn.py`; old snapshots stay.

## D401 — Process quotas in force

- **Question:** how the experiment share is handed out when no experiment runs, and how the agent pool picks among processes.
- **Choice:** proportional redistribution; limits rewritten at process start and when the running set changes; order human → least-loaded process by used/share → priority → age.
- **Rationale:** 11 §3.1 and §4.2 say this but leave the mechanism open.
- **Cost of changing later:** low: `src/kta/orchestrator/quotas.py`; WP6's agent step must call `order_queued`.

## D403 — Experiment protocols, batches and runs

- **Question:** what an experiment Protocol must contain and how runs are executed and recorded.
- **Choice:** params arms/runs_per_arm/cases/snapshot/profile/budget_per_run(+conditions, batch, max_concurrent); approval freezes; batch built and verified via `batch_envs`; one distinct solve loop per arm × case × rep; ExperimentRun per finished run; result Report with every run.
- **Rationale:** uses D345's record conventions; nothing is selected or dropped (50 §1.5).
- **Cost of changing later:** medium: the arm condition, pinned snapshot and batch are recorded but not yet read by retrieval or the agent step (WP6/WP8/knowledge change needed before WP12 results mean anything).

## D405 — Calibrate process triggers

- **Question:** when calibration refits the model, raises a T4, or re-measures hardware.
- **Choice:** refit after 50 new pairs, adopted only if hold-out error improves; T4 when |median signed error| > 15 % over ≥ 10 of the last 50 pairs of a family; T5 per hw_fact Gap; peaks when the HwModel is missing or older than the Target record.
- **Rationale:** 11 §1 lists the triggers without numbers; WP13 B1 found pre-feedback median error ~17 % and wide tails.
- **Cost of changing later:** low: `[calibrate]` config values.

## D406 — Improvement batches and attribution

- **Question:** when a batch closes and how judgements are attributed.
- **Choice:** experiment result, manual request or daily; rule 1 → verification, 2 → model, 4 → space; T3 per missing Template/Pattern Gap; system items wait for approval in the inbox.
- **Rationale:** 10 §2.3 gives the classes but no mapping from judgement rules.
- **Cost of changing later:** low: `src/kta/orchestrator/improve.py`.

## D407 — GB10 microbenchmarks v1 and profile codex-v2

- **Question:** which hardware parameters the first microbenchmark set measures and how, and how the new skills reach workers.
- **Choice:** L2 working-set sweep with an effective-capacity rule, dependent-load DRAM latency, single-CTA launch latency; codex-v2 adds diagnose-model and write-microbench.
- **Rationale:** WP13 B5 relates model error to the L2 boundary; WP7 already measures per-unit peaks.
- **Cost of changing later:** low: new benchmarks are additive; capacity rule is one function.

## D408 — WP11 gate check

- **Question:** how the workplan's "ordinary afternoon happens for real and is visible in the frontend" becomes a gate check.
- **Choice:** listed real records (microbench HwFact, ExperimentRun, calibrate T4, improvement Report), spans of four process lanes in one 4-hour window, and a screenshot.
- **Rationale:** the afternoon is defined by concurrent process activity; the frontend proof is a screenshot as in WP9.
- **Cost of changing later:** low: `tools/kta_gate/specific_wp11.py`.

## D422 — Error kind `cancelled` in core

- **Question:** the job-cancel follow-up asked for an ExecutionError kind such as `cancelled` and limited WP2's edits to `src/kta/execution`, but the error-kind literals live in `kta.core.errors` (`ExecutionKind`) and `kta.core.jobs` (`ErrorKind`, `ERROR_KINDS`).
- **Choice:** add `cancelled` to both literals (two additive lines in core) and to the spec's error-kind lists.
- **Rationale:** a distinct kind lets callers (WP6's runner watcher) tell a requested stop from an `infra` failure or a `timeout`; reusing `infra` would hide it in failure statistics.
- **Cost of changing later:** low; code that switches on error kinds should treat `cancelled` as terminal and not retry it.
## D373 — How an experiment arm's knowledge condition reaches the solve loop

- **Question:** WP11 pins a snapshot and a free-form condition per arm; nothing read them.
- **Choice:** the snapshot limits retrieval to its members. `condition.knowledge` = `pool` | `text` | `zero`; text and zero arms get no pool objects from retrieval, so the loop creates a T1 from the Case alone. Tasks carry `info["arm"]` and `info["batch"]`; the agent side (WP8) puts text knowledge or nothing into the context.
- **Rationale:** the three arms of docs/design/50 §1 differ only in what knowledge the agent and the solver see. Keeping the solver blind in text/zero arms makes the agent's work the measured variable.
- **Cost of changing later:** low: `solve_work.pinned` / `allowed_cards`; the key names are shared with WP8.
## D430 — How an experiment arm shapes an agent step

- **Question:** WP11 records each run's knowledge condition (from zero, text knowledge, Template pool) but nothing enforced it, so arms differed only in their label.
- **Choice:** the agents module reads `info["arm"]` from the task bookkeeping. `zero`: no input records of pool kinds, no `find-knowledge` skill, `kta query` not allowed. `text`: the same, plus the pinned snapshot rendered as Markdown in `inputs/knowledge/`. `pool`: unchanged. The condition is recorded in the context manifest and `AgentRun.arm`.
- **Rationale:** follows docs/design/20 §5 (same knowledge, different form) and the coordinator's brief. It departs from docs/design/10 §6 "skills are equal across arms" for `find-knowledge`, a skill about using the pool, in the arms that have no pool.
- **Cost of changing later:** low: `kta.agents.arms`. Retrieval inside `kta query` and in the solve loop (WP6/knowledge) must also honour the pinned snapshot; the agents module cannot enforce that alone.

## D325 Working answers to the open design questions (D-open-1 … 7)

- **Question**: 02 §12 lists seven open decisions; round 4 cannot wait for all of them.
- **Choice**: follow 02's own proposals as defaults (see D325 in `docs/decisions.md`); the largest is (5): template `effects` and the declarative hardware model are built now, before the GEMM golden path has run.
- **Why**: code stage (D320); no real records exist that a schema change would strand; building the golden path on the old `cost` schema would mean doing it twice.
- **Cost to change**: (5) reverting means keeping both `cost` and `effects` readers (upcast exists either way); (1)–(4), (6), (7) are data or protocol definitions, cheap to change.

## D440 — Schema versions and upcast on read (WP14)

- **Question:** round 4 changes record shapes, but records are immutable and their ID is the hash of the stored JSON.
- **Choice:** each kind has a tuple of upcasters (version = 1 + its length); a stored older body is upcast when read; `Record` keeps the stored body for hashing and transfer and shows the current shape as `body`.
- **Rationale:** readers see one shape, nothing stored is rewritten, `record_id` stays stable. The alternative (keep every old model and branch in every reader) spreads version checks through the code.
- **Cost of changing later:** low for new steps (append a function); WP15 appends its Template `cost → effects` step to the same tuple (note left for the coordinator).

## D441 — Level numbering 1–7 and the mapping from 0–5 (WP14)

- **Question:** D-T7 adopts the research frontend's 1–7; stored records and seed used the paper's 0–5.
- **Choice:** mapping `paper-0to5@1`: old n → n + 2 (0 graph → 2 graph … 5 microarchitecture → 7); nothing maps to 1 (workload). The seed is migrated in place by a recorded script; stored records are upcast on read.
- **Rationale:** the paper's 0 is "computation graph", the frontend's 2 is "graph"; every later level keeps its order. 
- **Cost of changing later:** low: a new named mapping and an upcaster; the seed migration is re-runnable.

## D442/D443 — Capability, CapabilityDecl and Realisation (WP14)

- **Question:** how to replace HwFeature (D-T8) without losing the probe machinery, and where code and SASS expectations live.
- **Choice:** Templates require abstract Capabilities; Targets declare them per record (newest wins) or support is inferred from probes of the ISA features that realise them; HwFeature stays as the ISA-level probe. Realisation = template × backend × ISA range with fragments, ISA constraints and expected SASS. Migrated seed realisations use backend `any`.
- **Rationale:** a new chip (or an NPU) adds declarations and realisations; Templates do not change. Declarations as separate records let a target "declare the capability" without rewriting anything.
- **Cost of changing later:** medium: Capability names are in every Template's `requires`; the ISA → Capability table is in the migration script. Backend `any` realisations should be split per backend once real families exist.

## D445 — Availability: what counts as unavailable (WP14)

- **Question:** whether `unknown` support hides a Template on a target.
- **Choice:** only `yes` makes a Template available (as before for HwFeature support); hidden ones are counted separately as "known, unavailable here" (`no`) and "unconfirmed here" (`unknown`).
- **Rationale:** a mechanism that cannot be shown to run should not enter search, but must stay visible; splitting the count tells which ones a probe could wake up.
- **Cost of changing later:** low: `capabilities.availability`.

## D446/D448 — One ladder, promotion and demotion (WP14)

- **Question:** D-open-1 (answered by D325's default) and how counter-evidence acts without a human.
- **Choice:** one ladder on the existing status values with evidence rules per step (≥ 2 replications; explanation or a CI; agreeing hold-out). Counter-evidence demotes one step as actor `system` (new store rule: system may move down, never up) and dependents become `stale`; rejection stays human-only. `review` only proposes promotions into the model, Γ/regime or a new Pattern.
- **Rationale:** reuses the stored status values, so no StatusChange is invalidated; demotion is invalidation, like `stale`, which is already a system action.
- **Cost of changing later:** low for thresholds (registry knobs once WP21 merges); medium for the actor rule (store/status.py, locked WP1 tests constrain checkers).

## D451 — GB10 capability declarations from documentation (WP14)

- **Question:** which Capabilities GB10 has, before any probe (D320).
- **Choice:** documented values with the source in each note; `warpgroup_mma` and `tensor_memory` `no`; `cluster_multicast` and `register_reallocation` `unknown` until probed; others `yes`.
- **Rationale:** the PTX ISA target requirements decide wgmma and tcgen05; multicast benefit and setmaxnreg on sm_121 are not documented clearly enough to assert.
- **Cost of changing later:** low: a new CapabilityDecl record (newest wins); a probe can also infer support.
## D460 — Template effects schema (transactions, storage, overlap)

- **Question:** how `cost` becomes `effects` (02 §3 sketch only).
- **Choice:** transactions name operation classes of the hardware model and carry per-transaction amounts; storage per executor scope; overlap between selectors with `when`, `inflight`, `trips`. Each physical transaction is declared once by the step that performs it.
- **Rationale:** no hardware number in templates (D-T2); "declared once" removes the WP5 missing-contraction issue without chain-wide transaction identities.
- **Cost of changing later:** medium: schema, the 37 seed Templates (one migration script), formulation blocks.

## D461 — Schema upcasts on read

- **Question:** records are immutable but the Template body changes.
- **Choice:** `KindSpec.upcasts` chain; stored bodies and IDs untouched; Template 1→2 maps cost terms to legacy `unit:<u>` transactions priced exactly as before.
- **Rationale:** one mechanism for every kind (WP14's Template change can register the next step).
- **Cost of changing later:** low; merging with WP14 needs one decision on version numbers (2→3 or one combined 2).

## D463 — HwModel split into class and instance

- **Question:** where structure (resources, operation classes) and numbers (peaks, limits) live.
- **Choice:** `HwArch` record for the class, `HwModel` schema 2 for the instance with per-value provenance; measured peaks override documented expressions.
- **Rationale:** D-T10 structure reuse; every value points to its source (R3).
- **Cost of changing later:** low–medium: consumers use `MachineModel`, not the records.

## D464 — L1 composition rule

- **Question:** how per-resource times, overlap statements and latencies combine (02 §5.1 gives ResMII/RecMII).
- **Choice:** max over busy times, pair bounds and recurrence bounds, plus serial and interaction terms; recurrence per executor with the smallest latency per group (optimistic).
- **Rationale:** each term is a lower bound of the steady state, so L1 ≤ L2 can be checked (WP16/WP17).
- **Cost of changing later:** low: one function (`kta.models.l1.compose`).

## D466 — UnitModel stays the calibration layer over L1

- **Question:** "one parameter table" (R1) vs the existing fitted UnitModel versions used by the orchestrator and T4 tasks.
- **Choice:** keep UnitModel's fitted eff/interactions as an overlay on the declared efficiency; composition is L1's.
- **Rationale:** no change to T4 outputs, calibration records or WP6; avoids two fitting paths now.
- **Cost of changing later:** medium: fold fitted values into HwModel instance versions with ModelVersion records (WP16 maintenance).

## D468 — Seed Templates migrated in place, versions kept

- **Question:** new content under the same `name@version` vs bumping versions.
- **Choice:** in place; a pre-WP15 store conflicts on re-import.
- **Rationale:** D325(5): no real data; avoids 147 chain-reference edits colliding with WP14.
- **Cost of changing later:** medium: bump versions and chain refs if a store with real seed records ever needs both.

## D474 — Upstream calibration contract as documented values

- **Question:** the GB10 instance needs numbers before KTA measures any.
- **Choice:** upstream-measured values labelled `documented`, with hash and commit; three placeholders `default`; CTA dispatch rate unmodelled.
- **Rationale:** honest labelling; KTA's own protocol has not run (D320).
- **Cost of changing later:** low: the first `kta.measure.peaks` run supersedes them with `measured` peaks.
## D480/D481 — Controls registry and its versions (WP21)

- **Question:** where the knobs of docs/design/12 live, how a value change is versioned, and how runs record it.
- **Choice:** definitions as package data in `kta.core` (layer 0) with typed accessors; values in force as ControlsVersion records + pointer `controls:current`; `refs.controls` on TaskCard, AgentRun, ExperimentRun; system versions when knobs are added.
- **Rationale:** one place for every knob that every layer can read; versions are evidence (immutable, with author and reason), so "which values did this run use" is a record lookup.
- **Cost of changing later:** medium: a different store for values touches `kta.store.controls` and the API; records already written keep their `refs.controls`.

## D482/D483 — Code reads knobs only from the registry (WP21)

- **Question:** the code had its own defaults (orchestrator `Settings`, AgentProfile policies, inbox constants, `[budget.quotas]`, `[agents] slots`), some differing from 12 (T4–T6 wall budgets).
- **Choice:** all read from the registry; 12's values win (T4 45 min, T5/T6 30 min); config keys that held knobs are refused with a pointer; thresholds that were not rows of 12 became rows (12 §7, a new section in the user's design document).
- **Rationale:** the user asked for "one tunable table"; two sources would let them disagree silently.
- **Cost of changing later:** low for values (a controls version); medium to move a knob back to configuration.

## D485 — Staged release defaults to S0 (WP21)

- **Question:** which stage a fresh store starts in, and where the gate sits.
- **Choice:** S0 (no agent calls) until the user writes a controls version changing `release.stage`; the gate is in the agent pool; manual `kta task lease` is not gated.
- **Rationale:** 01 §3 "no stage opens until we understand and have debugged the previous one"; a merged code base must not start spending agent time on its own.
- **Cost of changing later:** low: the knob's initial value, or `stages.yaml`. Note for merge: after this lands, `kta runner start` takes no task until the stage is S1 or later.

## D486 — Operator token only for settings writes (WP21)

- **Question:** the workplan asks for an operator token on settings writes, while design/30 §4 says one user and no login, and `/actions/*` have no token.
- **Choice:** the token guards `POST /controls` and `POST /release` only; `/actions/*` are unchanged.
- **Rationale:** follow the workplan without widening the change; the inconsistency is visible here.
- **Cost of changing later:** low: `check_operator` can guard the other write endpoints; the web page already sends the header.


## D560 — Language gaps admitted, not refused (WP20)

- **Question:** D-open-2: what the admit step does with something the expression language cannot state.
- **Choice:** a third Issue severity `gap`, assigned by the `admission` table in `src/kta/knowledge/learning.yaml`; gaps are counted and the record is admitted at `proposed`. Rows today: `bad_expression` for unsupported constructs (typos stay errors) and `undeclared_pattern_param`.
- **Rationale:** 02 §8.6 asks to count language gaps instead of refusing; `p.*` has no declared vocabulary (WP10a critique 7), so each use of a new `p.*` name is a vocabulary gap. The 61 such issues on the WP10a references (63 before WP15's effects) would otherwise be warnings to explain.
- **Cost of changing later:** low: one table row moves `undeclared_pattern_param` back to `warning`; typos in `p.*` names would then need explanations again.

## D561 — Admission rejects unexplained warnings (WP20)

- **Question:** how "warnings must be explained" is checked, and where the explanation lives.
- **Choice:** `warnings.yaml` at the root of the reference directory, keyed `<kind>@<file>:<where>`; an unexplained warning rejects the T2 (back to the same session, ≤ 2 retries) or refuses an import batch. `kta init --references` keeps returning warnings without refusing. The three WP10a `constraint_false` warnings are explained in new seed files.
- **Rationale:** a stable key the agent can copy from the rejection text; explanations stay with the deliverable, so a later reader sees why a chain violates an obligation.
- **Cost of changing later:** low: the key format is in `admission.warning_key`; the T2 check and the import share `admit`.

## D563 — System-level features by keyword table (WP20)

- **Question:** how coverage tells system-level features (02 §8.2) from scheduling moves in existing records without a schema change.
- **Choice:** regex table over unmatched-transaction descriptions plus `pattern_guess: system`; the decompose-kernel skill asks workers for the latter.
- **Rationale:** the 55 WP10a unmatched transactions are prose; a field on UnmatchedTx would need Reconstruction schema 3 and re-decomposition.
- **Cost of changing later:** medium: an explicit field (schema bump, upcast) and a migration of the seven references.

## D566 — Ablation design and coupling threshold (WP20)

- **Question:** which pairs to test, which design, and when a pair counts as coupled (02 §8.4).
- **Choice:** suspected pairs from glue constraints, Template relations and shared `signature.resources`; pairwise design up to 32 runs, else a 2-level fractional factorial ≤ 64 runs; coupled when |joint / product − 1| > 0.05 (registry knobs once WP21 merges).
- **Rationale:** cheap enough to run on every reproduced family; 5 % sits above the measurement noise band we expect but is not calibrated (X6 sets the noise band).
- **Cost of changing later:** low: table values; the resource suspect could read WP15 effects instead of `signature.resources`.
