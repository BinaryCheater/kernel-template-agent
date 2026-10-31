# kta.orchestrator

Layer 7. Specification: [spec/modules/orchestrator-cli.md](../../../spec/modules/orchestrator-cli.md). Built by: WP6 (solve, maintain); WP11 (the rest).

## Responsibility

Deterministic loops (solve, expand, learn, calibrate, maintain, experiment, summarize), task queue, budgets, completion checks and pending items.

## Public interface

- `step(loop, ctx) -> Transition` (`step.py`): pure; the state's work is passed in as `loop.facts` (D260).
- `machines.py`: `machine(id)` loads `machines.yaml`, the single definition of the solve, expand and task
  machines (D96); `Machine.edge(...)` refuses any transition the YAML does not list.
- `solve_loop.py`, `expand_loop.py`: the pure transition functions; `judge` applies rules 1-4 (D263).
- `solve_work.py`, `expand_work.py`: each state's work through the lower modules, returning facts.
- `learning_plan.py` (WP20): dry-run plans of a reference's intake → decompose → admit → reproduce → ablate → cover (D569).
- `runner.py`: `Runner.start(kind, subject, name)`, `Runner.advance(loop_id)` (loop lock, spans, record
  drafts, tasks; failures block the instance, D264).
- `tasks.py`: `TaskQueue` (create, lease with budget reservation, start, submit, check, recheck, retry,
  cancel, expire; D265, D266, D269); `done_checks.py` (D277); `outputs.py` (accepted outputs, D269).
- `budget.py` (reserve/settle over `loop:` > `process:` > `global`, D268); `inbox.py` (D270).
- `pipeline.py`: `Services` (one JobExecutor per process), family packages (D272), generate and build.
- `space.py`: verify_space items (structural paths and 4 samples) and reruns of failed checks.
- `control.py`: pausing `global`/`process:`/`loop:`/`task:` scopes (D364); `operations.py`: `Operations`
  (human task control, priorities, pause/resume, re-verification) in the API's `Orchestration` shape
  and `reverify_suspects` for the maintain process (D365, D368).
- `processes.py`: `run(svc, name)`, `solve_tick`, `maintain_tick`, `request_stop` (D273).

- WP11: processes `learn`, `calibrate`, `experiment`, `summarize` (`proc_*.py`, `improve.py`,
  `proc_common.py`); budget hierarchy and quotas (`quotas.py`); T5 check (`microbench_check.py`).
  See spec/modules/orchestrator-cli.md "WP11 implementation notes".

## Controls and staged release (WP21)

- `Settings.of(controls)` replaces `Settings.from_config`; the runner and the task queue read the controls in force at each step (`kta.store.controls.ensure`) and record the version (loop data `controls`, TaskCard `refs.controls`).
- `stages.yaml` + `release.py`: S0–S4; processes outside the stage skip ticks; the agent pool starts only admitted tasks; in S1 each attempt waits for a release.

## Gap decomposition and routing (WP18)

- `gap.py` + `routing.yaml`: per-round decomposition `T_meas − T_SOL = ε + δ + Δpre + L_rep + irreducible`, attribution order and the versioned routing table; `judge` writes a GapDecomposition per round (rules 1–6, terminal `closed`).
- `disciplines.py`: mechanisms per round, credit beyond the band, null rounds, rollback to the incumbent. `canary.py`: planted faults per summarize batch. Selection arms in `solve_work.selection_arms`; duplicate submissions in `TaskQueue.check`.
- See spec/modules/orchestrator-cli.md "WP18".

## Dependencies

- May import: `core`, `store`, `targets`, `execution`, `backends`, `knowledge`, `models`, `measure`, `formulation`, `checks`, `solve`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).

## WP19 (D544, D545)

`access.py`: `task_header` / `situation_of` (the one-screen case header), `record_pull`, `pull_log`, `pull_stats`, `plan_view`; `TaskRequest.header` goes into the card.

## WP22 (D585)

`inbox.py`: item `proxy:<provider>` while that provider's proxy breaker (ops pointer `proxy:status:<provider>`) is tripped.

## WP23 (D610–D615)

`stage_queues.yaml` + `stagequeue.py`: stages with WIP limits, backpressure, unblock-first order and measurement windows, used by `solve_tick`. `work_units.yaml` + `workunits.py`: the card work unit `<loop>:<role>` (knob `session.work_unit_rule`).
