# Clean baseline: formulation first, from the workload

Status: agreed with the user 2026-11-02 (discussion in the SYSDEBUG-2 session); implementation
in progress on branch sysdebug/2. "Exists" marks what was already in the code before.

## Principles (user, 2026-11-02)

- Templates contribute decision variables and tricks; they do not compose a whole program.
  The program is written by the agent.
- What is tested is the ability to formulate good solutions, not the ability to write a
  correct baseline from nothing. The agent may read TileLang documentation and examples.
  The baseline is not over-specified: let it come out as it does.
- The clean run offers no seed family, to exercise the whole flow. Seed families may exist
  in real work.
- A family is the formulation of one case (D713).
- No rule of "one change per round". Avoid circling on small points round after round;
  substantive changes (several related mechanisms if needed) are the default, a small one
  (e.g. one variable's range) only when the agent judges it the better move.
- Context management: the system keeps the state across rounds (code, formulation, what was
  tried and its outcome) and gives the agent only what is new; repeated exploration and
  repeated artifacts cost tokens and time.

## The flow

0. **Workload.** Case: semantics with a reference implementation, dtypes, layouts, shapes,
   tolerance. The system adds reference timing, bounds, hardware facts and the Template
   cards relevant to the workload. No family is offered in the clean run.
1. **Round 1.** Retrieval finds no family of this case -> T1 "write a baseline for this case
   and its formulation" (exists as the no-family T1 path; prompt updated). The system
   verifies the delivered family (validate, space verification, proof gate), then solves it.
2. **Every round with a family.** Assemble -> default FormulationPlan -> projection MIP
   (finite menus one-hot, menu products and table terms tabulated, ceil columns, outer
   enumeration; exists in `kta.solve.projection`, now called by the loop). Enumeration is
   the fallback, with its reason recorded. Build the top `compile_count` (8), check,
   measure, judge by gain per shape.
3. **Next T1.** The agent gets the delta since its last step: measured gain per shape,
   predicted vs measured, the new gap; it changes the family as it judges (one or several
   mechanisms), working on the current family already placed in `out/`.

## Context management

| Item | Rule |
|---|---|
| Resume | T1/T6 of one case share a session across rounds (`case_role`, D614, exists); fresh on context > 60 %, abnormal exit, same failure twice, force retry (exists) |
| Never shared | across cases; kernel writer vs model diagnosis (exists) |
| Resumed step | the step message carries only the delta (results of the last delivery), not the full inputs again |
| Fresh step in a unit with history | a handoff written by the system from records: case state, family lineage with each version's change, what each round measured |
| Workspace | `out/` is prefilled with the current family; the system records the diff of each delivery against its parent |
| Monitoring | per agent step: calls, input tokens, cached share, resume or fresh and why |

## Display (case page, per round)

- formulation: family and why; exported constraints (source step, expression, meaning);
  the family diff against the previous version.
- compiled + solved: method and reason, MIP status/gap, space -> feasible, shortlist
  (assignment, predicted us, bottleneck unit), which were compiled, proof gate.
- agent work: runs with tokens, cached share, resume/fresh.

## Run

SYSDEBUG10: a fresh store, GEMM fp16 case, no seed family imported; prepared and stopped
before the first step, started only on the user's go-ahead.
