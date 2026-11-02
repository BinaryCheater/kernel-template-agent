# Clean baseline: formulation first, from the workload (plan, for user review)

Status: plan only, nothing below is implemented unless marked "exists". User request,
2026-11-02: formulation runs from the first round; whatever is missing is filled in; a
case can start from a pure workload, build its own baseline form and optimise it.

## What exists today (checked in code)

| Piece | Where | State |
|---|---|---|
| Chain enumeration from the Template pool for a case, bounded by SOL | `kta.solve.chains.enumerate_chains` | exists, not called by the solve loop |
| Assemble a formulation (variables, constraints, effects) from a chain | `kta.formulation.assemble` | exists, used every round |
| Finite-menu linearisation: one-hot menus, menu products and table terms tabulated over joint menus, integer ceil columns, outer enumeration for what fits neither; objective = L1 projection (T >= busy per resource, pair and recurrence bounds, walls) | `kta.solve.projection.compile_and_solve` | exists (WP17/WP19), not called by the solve loop |
| Deterministic FormulationPlan (F1-F7), diff and score of an agent's plan | `kta.solve.plan` | exists, not called by the solve loop |
| Enumeration over the family's declared domains | `kta.solve.enumerate` | what every round uses now |
| Code generator | hand-written per family (`gen.py`) | the one piece a pure workload does not have |

So the round-1 "Formulation" step today only picks a stored family; the MIP path is
skipped because the plain MIP export refuses products of variables, and the projection
compiler that handles them is never given a plan.

## Target round (every round, round 1 included)

1. **Workload -> chain (F1).** Round 1 without a family: `enumerate_chains` over the
   Template pool for the case, ranked by bound; keep the best few chains. With a family in
   the case's own lineage (D713): its chain is the default; the chain enumerator still runs
   and is shown, so a better chain can be proposed.
2. **Chain -> formulation (F2-F6).** Assemble; build the default FormulationPlan
   (menus, outer variables, table terms, glue, dispatch). Record the plan.
3. **Export.** The exported problem is stored and shown: variables and menus, every
   constraint with its source step and meaning, the objective atoms, and which products
   were tabulated / which variables went outer.
4. **Transform.** Deterministic: the projection compiler's linearisation (above). Agent:
   a T1 may change the plan (chain step, menus, glue, table terms); its diff against the
   default and its citations are scored (`score_plan`) and shown.
5. **Solve.** `compile_and_solve` per shape bucket (MIP, status and gap). Enumeration is
   kept as the comparison arm on small spaces and as the fallback, always with its reason.
6. **Baseline code.** If the chain has a generator (a family in the lineage, or a library
   one taken as a starting point), build the top `compile_count` (8) candidates, check and
   measure as now. If there is none, the round still delivers the formulation, the
   predicted best region and the bound, and the T1 goal becomes "write the generator for
   this plan" (F7) with the plan as input; the next round builds and measures it.
7. **Judge and next round** as now (gain per shape vs the previous round and the reference).

## Display (case page, per round)

- formulation column: chain source and reason per step (which cost it addresses,
  alternatives not taken), exported constraints/objective, transforms, plan diff.
- compiled + solved column: method and reason, MIP status/gap, space -> feasible with
  per-constraint pruning, the shortlist (assignment, predicted us, bottleneck unit),
  which 8 were compiled, proof gate.
- after measuring: predicted vs measured per point and rank correlation.

## Steps

| # | Step | Size |
|---|---|---|
| 1 | Solve loop calls `default_plan` + `compile_and_solve` per bucket; enumeration as comparison/fallback; record method, reason, status, gap, pruning | small-medium |
| 2 | Store the exported formulation and plan per round; case page shows them and the shortlist | medium |
| 3 | Round 1 without a family: chain enumeration -> plan -> solve; T1 "write generator for this plan" | medium |
| 4 | Per-step reasons and alternatives in the chain (from the enumerator's ranking and the Templates' cost notes) | small |
| 5 | Clean store SYSDEBUG10: GEMM case with no seed family offered (pure workload) and, as a second arm, with the TileLang seed as a starting point; compare | run |

Open questions for the user: whether the no-generator round should also ask the agent for
a generator at once (one T1) or first show the plan; whether the seed family stays
available as a starting point in the clean run or the run starts with none.
