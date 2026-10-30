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
