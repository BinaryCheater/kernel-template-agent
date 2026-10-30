# WP6 delivery: orchestrator and cli

Branch `wp/WP6`, cut from `integration` @ `90aa678` (WP0–WP5, WP10a, WP13 merged). Code commit of the evidence runs: `3a5fece8ffc342cc5a60b45af5ff3272549de2d8` (clean). Dev store: `~/kta-data/dev/WP6/kta.db` on the control host (this Mac); command outputs under `~/kta-data/dev/WP6/evidence/`. WP6 is not record-backed (D84); the record IDs in `deliveries/WP6.evidence.json` are listed anyway. The first end-to-end solve is WP7's (D94).

## What was done

**State machines (`src/kta/orchestrator/machines.yaml`, D96).** WP6 now owns the file. It remains the single definition, and the frontend's format is unchanged. The `notes` now record how each path the design left open was settled:

- **Solve loop:**
  - A waiting loop resumes at `retrieving` after T1, T2 or T6 (the pool changed). After T4 it resumes at `solving` (the model changed). The draft's edge `waiting -> judging` is removed. Each resume starts a new round; at the round limit the loop is escalated instead (D261).
  - New edges: `solving -> waiting` (T1) when no candidate is feasible, and `building -> waiting` (T6) when nothing builds (D262).
- **Expand loop:**
  - A missed reproduction epsilon goes to `gap` and creates T1. Its acceptance resumes at `reconstructed`.
  - After extraction tasks, a new T2 re-matches the chain (D267).
- **Task lifecycle:** new state `suspended` (a checker crashed; does not count as a retry) and the edge `leased -> queued` (lease lost) (D266).
- **Work failures** are not transitions: the instance stays in its state, marked blocked (D264).

**Pure transition function (D260).**

- `kta.orchestrator.step(loop, ctx) -> Transition` dispatches to `solve_loop.step` and `expand_loop.step`.
- Each state's work runs first, in `solve_work` / `expand_work`, and is passed in as `loop.facts`.
- `step` reads only the loop row, the facts and `ctx.config`. It returns a target state (or a hold), the YAML kind and rule, data updates, task requests and record drafts.
- `Machine.edge()` refuses any transition the YAML does not list, and the runner refuses task types the edge does not list.
- **Judging (D263):** rules 1–4 are applied in order, and every judgement writes a Report titled `judgement` with the fields WP9a reads (D169).
  - Rule 1 (failed check) creates T6 with the failed (Check, Candidate) pairs.
  - Rule 2 creates T4 when Spearman rho < 0.5 over at least 3 measured candidates, or median relative error > 0.3.
  - Rule 3 accepts and writes one Solution per bucket. It requires every bucket to be no slower than the strongest baseline beyond 3 × the noise floor, or the model to explain at least 80% of the gap to the bound.
  - Rule 4 creates T2 when a Reference of the case's Problem exists, else T1.
  - A result above 95% of the bound, or more than 1.3× faster than the baseline, is flagged as suspicious.
  - The thresholds are in config `[orchestrator]`.

**State work (`solve_work.py`, `expand_work.py`).** Each solve state's work is a real call into the lower modules:

| State | Work |
| --- | --- |
| `new` | `case_view`: Case, Problem, Contract, baseline |
| `bounded` | `models.bound` per bucket; writes Bound records |
| `retrieving` | `retrieve_families`. With none, `nearest_chains` + `reuse_plan` (or `retrieve_templates`) go into a `retrieval` Report and become the T1 input |
| `solving` | `assemble`, `solve`, `record_solve` |
| `building` | generate → `backend.build` through the Executor → `sass_check` → `rerank` |
| `checking` | `run_full` |
| `measuring` | `Measurer.measure` |
| `judging` | Collects the per-bucket facts |

- **Expand:** the freeze check is real. Building, measuring, reproducing and ablating references raise NotSupported (WP11).
- **Services and families:** one `Services` per process holds a single JobExecutor (D184), shared with the task checks. Family packages are materialized from the Family record's `files/<path>` blobs, or else from `<seed_dir>/families/<name>` (D272).

**Runner (`runner.py`).** `Runner.start(kind, subject, name)` and `Runner.advance(loop_id)`:

- `advance` takes the loop lock, runs the work, then `step`, then writes the result, then releases the lock.
- It writes one `loop_step` span per state visit, with trace = loop ID, and the loop `data` keys of D169.
- Record drafts are written in order, linked by index.
- Created tasks are written with `waiting_on`, `waiting_tasks` and `waiting_type`.
- **Blocking (D264):** NotSupported, schema and checker errors block the instance in place. Infrastructure and timeout errors are retried up to 3 times first. Starting the loop again clears the block.
- **Attention limit:** at 10 inbox items, transitions that would create tasks are deferred.
- **Restart:** a fresh process continues from the stored state (tested).

**Task queue (`tasks.py`, `done_checks.py`, `outputs.py`).**

- Cards are immutable TaskCard records with `profile = "unassigned"` until WP8 (D271).
- **States:** every state change is a compare-and-set on the ops row, using lifecycle names (D265) and checked against the YAML.
- **Lease:** reserves the card budget in the scopes `loop:` > `process:` > `global`, all or none (D268). A refusal keeps the task queued.
- **Submit:** writes a Submission, with each file as a blob.
- **Check:** runs the fixed done-checks of the task type (D277) and writes a Report titled `done_checks` (D276).
- **Acceptance** writes the outputs into the pool and a `task_result` Report (D269). Outputs per type:
  - T1/T6: Family and Chain, with the files as blobs;
  - T2: Reconstruction, Chain and Gaps;
  - T3: a Template via `propose_template`;
  - T4: a model version.
- **Rejection:** goes back to queued, or to escalated after 2 retries.
- **Suspension:** a checker failure suspends the task.
- **Other operations:** cancel and retry by you write a HumanAction. Expiry returns lost leases to the queue and expires silent running tasks.
- **Human/CLI submission path:** `kta task submit --dir D [--check]` (D269). `kta task create` starts a task yourself (D278).

**Budgets and inbox.**

- `budget.py`: reserve and settle over the scope chain, `view` (budget, used, reserved, remaining), and `[budget]` config with the quotas of docs/design/11 §3.1.
- `inbox.py` derives items from stored state (D270):
  - escalated or suspended tasks;
  - escalated or blocked loops;
  - suspicious results;
  - a done-check failing 3 or more times;
  - 3 judgements in a row with rule 2;
  - a budget at 80%.

**Processes (`processes.py`, D273).**

- `solve`: checks submitted tasks of solve loops, then advances every solve loop once.
- `maintain`: expires leases and prunes spans older than 90 days.
- `kta process start <name>` runs in the foreground, one instance per name (ops lock). `kta process stop` writes a timestamped stop request.
- The other four processes raise NotSupported (allowed simplification).

**CLI (D274, D278).**

- **Management:**
  - `kta case add`, `kta reference add`, `kta solve <case> [--steps]`, `kta expand <reference>`;
  - `kta loop list|show|step`, `kta process start|stop`;
  - `kta task list|show|create|submit|check|recheck|cancel|retry`;
  - `kta inbox`, `kta budget`, `kta coverage`, `kta export`.
- **Agent-facing (no auth yet):** `kta show`, `kta query pool|tmpl|chain`, `kta why`, `kta card`, `kta note`, `kta submit`, `kta validate-template`, `kta validate-family`.
- Every command supports `--json`, and the ones that act write a `cli` span.

**Real dev run (`KTA_ENV=dev KTA_WP=WP6`, commit `3a5fece`; claims 1–4).**

- **Solve:** `kta init` imported the seed, and `kta case add` added an FP16 GEMM case (1024³, gb10). `kta solve` went `new -> bounded` and then blocked: NotSupported, no HwModel record for gb10. No store has an HwModel yet (WP2, WP3 and integration were checked). The inbox lists the blocked loop.
- **T3 accepted:** a T3 created with `kta task create` was submitted through the CLI path.
  - It leased with a reservation in `global` (agent_min 600).
  - It passed delivery_note and validate_template, wrote the Template (actor.task set), the task_result and done_checks Reports, and settled the reservation back to 0.
- **T3 rejected:** a second T3 was rejected twice (a schema error, then `bad_expression` in `c_aligned`), going back to queued each time, then cancelled (HumanAction).
- **Expand:** an expand loop on a CUTLASS Reference passed the freeze check and blocked in `frozen` (NotSupported, WP11).
- **Processes:** `kta process start solve --ticks 2` and `maintain --ticks 1` ran.

**Tests and gate.**

- 58 new unit tests:
  - `test_orchestrator_step.py` (28): YAML equality for both machines, every judging rule, resume, round limit and expand branches;
  - `test_orchestrator_tasks.py` (13): every lifecycle edge, read from ops events, budgets, T1/T2/T3 checks;
  - `test_orchestrator_runner.py` (8);
  - `test_cli_orchestrator.py` (5);
  - `test_gate_wp6.py` (4).
- WP6 gate check `wp6` (D275): the named transition tests passed, and every judging rule number in machines.yaml has a passing `test_judging_rule_<n>_*`.

## Fixtures and simplifications

- No `# kta:fixture` added to `src/`. Inherited, unchanged: `src/kta/store/bench.py` (`_body`, D156).
- Allowed simplification used: of the six processes only `solve` and `maintain` run. `learn`, `calibrate`, `experiment` and `summarize` raise NotSupported (WP11).
- Not implemented (they raise NotSupported):
  - expand-loop work after the freeze check: reference build, measurement, reproduction and ablation (WP11);
  - the T5 done-check `microbench_repeats` (the task is suspended);
  - agent-facing `instantiate`, `build`, `sass`, `run`, `src` (WP8).
- Test data, not knowledge:
  - The real-run T3 Template `async_transfer_wp6_cli_check` is a renamed copy of the seed's `async_transfer`, used only to exercise the path. Its delivery note says so, and it is now a proposed Template in the WP6 dev store only.
  - Tests use `TEST_HW` and the WP5 test family (`tests/acceptance/WP5/gemm_family`).
- No agents exist: T1–T6 are created, queued and visible, and completed only through the human/CLI path. No agent output is faked.

## Known issues

- **The building, checking, measuring and judging work has not run against real data yet.** It needs an HwModel record (measured peaks) and a real family, and neither exists. `bounded` blocks today. These handlers are written against the WP3/WP5 APIs, but only `new`, `bounded` (with a test HwModel), `retrieving` and the waiting states are exercised by tests. WP7's first run is where these paths get debugged.
- **Generator contract.** `pipeline.generate` expects `gen.py` to return a GeneratedKernel or a mapping with `files`, `entry` and `launch` (D272). The WP5 test family returns only files and markers, so `verify_space` rejects it, which is tested. WP7's family must return `launch`.
- **One family per loop instance.** The best retrieved family is used, and the others are ignored.
- **Builds are not linked to candidates.** WP2's backend writes no `refs.candidate` on Build, which D169 expects. The loop keeps the candidate-to-build map in loop data instead.
- **verify_space** uses the first bucket's shape only, and visits at most 10⁵ feasible points when looking for structural paths. Its StatusChange subjects are empty because the Family record does not exist until acceptance.
- **Inbox.** Worker/GPU unavailability is shown only through blocked loops (there is no worker listing). "Model error repeatedly large" means the last 3 judgements were rule 2.
- **Processes.** The solve process checks submitted tasks of solve loops only. Human tasks (`kta task create`) and expand-loop tasks are checked by `kta task check` or `kta task submit --check`. No new pool snapshot is made after acceptance (the learn process, WP11).
- **Human/CLI submissions book no agent time** at settlement (D268). Reservations still gate the lease.
- **Expand T1 after a missed epsilon** does not list the new Gap in its inputs (the Gap ID is only in loop data).
- **CLI.** `kta coverage` prints a short summary; `--json` gives the full report. Commands that only read (`list`, `show`, `inbox`) write no span.
- **Concurrency.** `Runner.advance` holds the loop lock for up to an hour per step. A crashed holder blocks that instance until the lock expires (`LOCK_S`).

## New decisions

D260, D261 (needs user review), D262 (needs user review), D263 (needs user review), D264, D265 (needs user review), D266 (needs user review), D267 (needs user review), D268, D269, D270, D271, D272, D273, D274 (needs user review), D275 (needs user review), D276, D277, D278.

Round 3: D372, D373 (needs user review), D374, D375, D376, D377, D378.

Follow-up (integration wiring): D360 (needs user review), D361, D362, D363, D364 (needs user review), D365, D366, D367, D368 (needs user review), D369, D370 (needs user review), D371.

## Needed changes

- **store (WP1 owner), additive, already made:** `SqliteOps.update_task` (compare-and-set of the lifecycle state), `list_tasks` and `list_loops`. These are recorded in `spec/modules/core-store.md` and the store README (D265). The `Ops` Protocol is unchanged.
- **WP9a (api/web):**
  - Ops `tasks.state` now holds lifecycle names (D265), so `TASK_STATE` and the `escalated` heuristic in `src/kta/api/runs.py` can go.
  - Rejections are Reports titled `done_checks` (with `data.passed` false), not Check records (D276).
  - The inbox can call `kta.orchestrator.inbox.inbox` (D270) instead of its own two triggers.
  - `machines.yaml` gained the `suspended` state and several edges.
- **WP2 (backends), optional:** have `Build` carry `refs.candidate` when the caller knows it (D169).
- **WP7:**
  - an `HwModel` record with measured peaks for gb10, written before `kta solve` can pass `bounded`;
  - a family whose generator returns `launch` and `entry` (D272);
  - a baseline Measurement with `refs.case` and `refs.baseline`, read by judging.
- **WP8:**
  - the agent-facing `instantiate`, `build`, `sass`, `run` and `src` commands;
  - task-token authentication;
  - a runner that leases with `agent:<n>`, calls `TaskQueue.start`/`submit` and replaces the `unassigned` profile;
  - the T5 microbenchmark check format (or WP11).
- **Spec edits made in this WP:**
  - `spec/modules/orchestrator-cli.md`: implementation note on `step` and its files, done-check names, CLI additions and deviations;
  - `spec/modules/core-store.md`: the three additive ops methods.

## Pending real runs

WP6 is not record-backed (D84); its real-run acceptance is part of WP7 (spec/modules/orchestrator-cli.md "验收"). Still owed, paused by D320:

- **The real solve loop run (WP7).** `kta case add gemm_fp16` (after `kta target probe`, for the baseline image), the measured HwModel, the GEMM family record, then `kta solve` to `accepted` or to a recorded judgement on spark. This is the first real run of these:
  - the `solving`, `building`, `checking`, `measuring` and `judging` work, including the small sanitizer shape (D360), SASS expectations per item (D361) and the baseline measurement (D362);
  - `verify_space` and `failing_cases_pass` (T1/T6 done-checks).
- **Restart in the middle of the run (WP7).** Kill the solve process mid-run, restart it, and check that it continues from the stored state.
- **One agent step through `kta runner start` (WP8).** It needs an approved profile (`kta profile approve`), a built batch, credentials and the agent host. From lease through `run_step` to check or abandon, then a retry with a note (D365, D369).
- **Re-verification of a real suspect Solution (WP9b, D368)** and a pause/resume from the frontend (D364).
- **An experiment batch (WP11)** with a pinned snapshot and the three knowledge arms through `kta runner` (D372, D373); a cancel during a real agent step once WP2's `cancel` is merged (D374).
- **Rerun of the WP6 dev evidence after the merges.** The runs in `deliveries/WP6.evidence.json` were made at `3a5fece`, before `integration` @ `868885b` and @ `528caab` were merged. They were not repeated (D320). All of them were local; none used spark.

## Follow-up: integration wiring

Second round, after `integration` @ `528caab` (WP6, WP7, WP8 and WP9b merged) was merged into `wp/WP6`. Code only (D320); nothing below has run on real data. Decisions D360–D371.

**Wired.**

- **From WP7:**
  - `work_checking` passes `sanitizer_shape`: the Contract's first non-trivial edge shape (D360).
  - SASS expectations are filtered by their `when`, evaluated per chain item over its bindings (`pipeline.sass_for`, D361). This is used in building and in the full check.
  - `work_measuring` times the Case's baseline Report per bucket with `measure_baseline` (D362).
  - `Case.target` is the Target name (D287, D363).
  - `kta case add` reads the seed case format, by path or seed name, e.g. `kta case add gemm_fp16` (D363).
- **From WP8:**
  - `kta agents ...` is registered from `kta.agents.commands` (D319).
  - The agent runner process `kta runner start|stop` (D369) leases as `agent:<slot>`. It calls `run_step` with the last rejection's `RetryNote` and the human note, then checks a submitted task or abandons one left running (`TaskQueue.abandon`).
  - New commands: `kta profile approve` (`approve_profile`) and `kta unit close` (`close_work_unit`).
  - New cards name the approved `current_profile` (D366).
  - `kta instantiate|build|sass` work on a Family record (D371).
  - CLI client mode when `KTA_TASK_TOKEN` is set (D370).
- **From WP9b:**
  - `TaskQueue.create/cancel/retry/recheck/set_priority` take the human Actor and write one HumanAction as that actor (D365).
  - Human-created tasks get the work unit `human:<id>`.
  - A retry note is handed to the next agent step.
  - Priority updates use the additive `Ops.set_task_priority` (D367).
  - Pause/resume of `global`, `process:`, `loop:` and `task:` is honoured by process ticks, loop steps and leases (D364).
  - A suspect Solution, or a record a Solution rests on, is re-verified by a fresh solve loop. The maintain process does this for suspect StatusChanges, and `request_reverify` on request (D368).
  - `kta.orchestrator.operations.Operations` implements the API's `Orchestration` calls; WP9b can replace `Unwired` with it.
  - `inbox(svc)` is in `kta.orchestrator.inbox`.
- **Tests:**
  - `tests/unit/test_orchestrator_wiring.py` (8): human operations, profile on cards, pause, retry note, abandon, re-verification, SASS `when`, sanitizer shape;
  - 4 more in `test_cli_orchestrator.py`: client mode, the registered commands, seed case add, runner selection.
  - Full unit suite: 487 passed.

**Still NotSupported or open.**

- **Stopping an agent on cancel.** A cancel during an agent step marks the task cancelled, but the agent job runs to its end: the Executor (WP2) has no job cancellation. The runner then leaves the task cancelled. This needs `JobExecutor.cancel(job_id)` (WP2) and a watcher in `kta runner`.
- **Client mode** works only for `kta note`. Show, query, validate, build and submit need token endpoints in the API (WP9b, D370).
- **API-side calls.** `Operations.approve_profile`, `verify_batch` and `verify_task_token` raise NotSupported naming the `kta.agents` function: the orchestrator (layer 7) may not import agents (layer 8), so the API calls `kta.agents` directly.
- **`kta run` and `kta src`** remain NotSupported (D371).
- **Re-verification** of kinds that no Solution rests on is marked unsupported (WP11).
- **The agent runner** runs one step at a time per process. Several slots mean several `kta runner` processes, but the lock name is one per host; adding per-slot locks is a small change.
- `expand_loop.py` and `expand_work.py` were left to WP10b, unchanged.

**New decisions:** D360 (needs user review), D361, D362, D363, D364 (needs user review), D365, D366, D367, D368 (needs user review), D369, D370 (needs user review), D371.

**Needed changes (follow-up).**

- **WP2:** cancel a running Executor job.
- **WP9b:** replace `Unwired` with `Operations`, plus the three `kta.agents` calls. Add token endpoints for the agent-facing commands (D370).
- **Additive changes made in this follow-up:** WP1's `Ops` Protocol and `SqliteOps` gained `set_task_priority`, recorded in `spec/modules/core-store.md` and the store README.

## Round 3

After `integration` @ `0475dea` (WP6 wiring, WP9b, WP10b and WP11 merged) was merged into `wp/WP6`. Code only (D320); nothing below has run on real data. Decisions D372–D378. I sent the arm key to the WP8 session; no reply yet.

**Wired.**

- **Agent queue order (WP11, D372).** `kta runner` picks the next queued task with an approved profile in `quotas.order_queued` order.
- **Task batch (D372).** A task runs in the batch from its bookkeeping (`info["batch"]`), else `--batch`, else `[agents] batch`.
- **Experiment pinning (WP11, D373).**
  - Solve-loop retrieval honours `loop.data.experiment.snapshot`: only snapshot members reach the solver.
  - It also honours `experiment.condition.knowledge` (`pool` / `text` / `zero`). Text and zero arms get no pool objects, and their T1 is created from the Case alone.
  - Tasks of experiment loops carry `info["arm"]` = {name, condition, snapshot, protocol} and `info["batch"]` when they are created. This is the key I sent to WP8.
  - An unknown condition, or a snapshot that is not a PoolSnapshot, blocks the loop.
- **Per-slot runner locks (D375).** `kta runner start --slot N` holds `process:agent:<N>`, so one host can run several slots. `kta runner stop` stops them all.
- **Cancel stops the agent (D374).** The agent step's Executor calls go through a `TrackingRunner`. A watcher thread cancels the step's jobs with `cancel(job_id)` when the task is cancelled.
  - This is coded against WP2's planned `JobExecutor.cancel(job_id) -> None`, which is not merged yet (`wp/WP2-cancel` has no commits).
  - Until then the watcher logs, and the step output says `cancel_unsupported`.
  - It is tested with the stand-in `tests/fakes/executor.py` (`RecordingExecutor`). Its contract test compares its methods with `JobExecutor` and `JobRunner`, and compares `cancel` with the real one once it exists.
- **WP10b, references (D376).** `kta reference add` accepts a `problem` body, writes it and links it.
- **WP10b, coverage (D377).** `kta coverage` prints one line per report key.
- **Re-verification (D378).** A suspect Bound that no Solution rests on is recomputed and compared, and the result is written as a Report. Other kinds without a Solution stay marked unsupported.
- **Tests.** `tests/unit/test_orchestrator_round3.py` (8 tests) and `tests/contract/test_executor_contract.py` (3). Full unit suite: 538 passed.

**Still NotSupported or open.**

- **Agent cancel** works only once WP2's `JobExecutor.cancel` lands. The watcher polls every 5 s.
- **The text-knowledge arm** reaches the solver as "no pool objects". Rendering text knowledge into the agent context is WP8's part.
- **Re-verification** of Templates, Families, Checks and Measurements that no Solution rests on remains unsupported.
- **WP11's `PINNING_NOTE`** in `proc_experiment.py` still says that retrieval does not read the pins. It is now outdated, but the file is WP11's.

**Fixture added:** `tests/fakes/executor.py` (`RecordingExecutor`, test double for the Executor boundary). `src/` is unchanged in this respect: no `# kta:fixture`.

**New decisions:** D372, D373 (needs user review), D374, D375, D376, D377, D378.

## Gate output

Code stage (D320), after the integration wiring at `integration` @ `528caab`: `KTA_DATA=~/kta-data scripts/gate --stage code WP6` on the control host.

```text
gate WP6 --stage code: PASSED (22 pass, 0 warn, 0 fail)
  PASS ruff format
  PASS ruff check
  PASS import layers
  PASS residue
  PASS test doubles
  PASS error handling
  PASS print
  PASS subprocess outside Executor
  PASS secrets
  PASS acceptance lock — 7 locked files unchanged
  PASS pyright
  PASS unit tests — 487 passed
  PASS contract tests — 5 passed
  PASS fake coverage — 1 fake(s) covered
  PASS registry (kta doctor, dev)
  PASS delivery note
  PASS fixture declarations — 1 declared
  PASS decision log — 292 decisions
  PASS skipped tests — none
  PASS evidence — 5 claim(s)
  PASS WP6 transition tests — 4 tests passed
  PASS judging rules tested — rules [1, 2, 3, 4] each tested
```

Earlier runs: the code stage after merging `integration` @ `868885b` (22 pass, 377 unit tests); the full gate at `77ca05c` before any merge (22 pass, 347 unit tests).
