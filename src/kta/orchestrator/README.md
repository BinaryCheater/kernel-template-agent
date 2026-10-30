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

## Dependencies

- May import: `core`, `store`, `targets`, `execution`, `backends`, `knowledge`, `models`, `measure`, `formulation`, `checks`, `solve`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
