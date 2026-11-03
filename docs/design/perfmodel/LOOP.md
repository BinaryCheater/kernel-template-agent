# The loop

Read this at the start of every round. It is the whole working method of the GB10 performance-model programme ([28](../28-gb10-programme.md)); nothing outside it is required reading for a round except `STATE.md`, the top of `error-table.md`, and the inventory row you are working on.

## The seven products — nothing else is produced

| # | Product | File | Changed by |
| --- | --- | --- | --- |
| 1 | Mechanism inventory | `docs/design/29-mechanism-inventory.md` | the round that measures a mechanism: its status row only |
| 2 | Parameter table | `src/kta/models/gb10sim/gb10.json` (every value: source, status) | a probe result or a declared fit (§4), never by hand |
| 3 | Machine description | `src/kta/models/gb10sim/machine.yaml` | adding or changing a mechanism |
| 4 | Simulator | engine, front ends, attribution, what-if under `src/kta/models/gb10sim/`, `tools/kta_perfmodel/` | adding or changing a mechanism |
| 5 | Probe set | one directory per probe: source, batch record, measured values, predicted values, tolerance | each round adds one; every probe is a permanent regression |
| 6 | Test set + scorecard | kernels (SASS, launch config, measured time, counter values, cache state) and `error-table.{md,json}` | the measurement and table scripts only |
| 7 | State | `STATE.md`: round reports, open questions with experiment counts, frozen parameters, fit ledger | the end of every round |

Notes, hand-over files, reports elsewhere: not allowed. Anything worth keeping goes into one of the seven. Decisions go to `docs/decisions.md` as usual.

## One round = one GPU batch, six steps

A probe answers one question; a round carries **as many probes as are ready**, in one GPU batch. Writing a probe is minutes of work when the upstream pattern exists (`agentic/tilelang/microbench/`, B1–B40); do not spread ready probes over several rounds. The first round therefore probes every `assumed` or `absent` inventory mechanism that a test-set kernel uses, in one batch, and updates all of their parameters together; later rounds are the disagreements that come out of the scorecard.

| Step | Who | What |
| --- | --- | --- |
| 1 | script | `scripts/perfmodel-table --top` prints the scorecard summary, the top rows with attribution and budget flags, and the fit-ledger line |
| 2 | agent | pick the mechanisms by the selection rule (§3) — all that qualify, not one; for each, write the probe (copy the upstream pattern) and its entry block (mechanism, why now, two or more hypotheses with the numbers each predicts over the sweep, the engine's replay prediction) |
| 3 | script | `scripts/perfmodel-measure <batch>` runs the batch on spark under the protocol (clock calibration, canaries, declared cache state, timing without the profiler, counters only from the profiler), validates it, and imports it; a batch without the entry block and the prediction is refused; an invalid batch is kept but never imported |
| 4 | agent | compare measured with each hypothesis; change the parameter table, the description or the engine accordingly |
| 5 | script | `scripts/perfmodel-table` reruns every probe and every kernel; the changed rows must improve and no other row may worsen beyond tolerance |
| 6 | agent | append the round report (§6) to `STATE.md`, update the inventory row, commit |

Rounds do not overlap. A round that cannot finish its step writes the report with "stopped at step N: <reason>" and ends. Time is spent on the GPU batch and on interpreting it, not on infrastructure: a script that does not exist yet is a thin wrapper over `scripts/remote-run` plus the checklist, written once, not a project.

## 3. Selection rule — how the next mechanism is chosen

The inventory lists every mechanism in advance with its status (measured / assumed / absent / partial). The scorecard says which of them currently costs accuracy. The rule:

1. Take the scorecard row with the largest absolute error among rows that are **valid**: counts check passed (R1) and cache state declared. **In the same round, also** take the variant group with the most inversions and its worst-inverted pair: the mechanism that differs most between the two replays (by attribution) is selected like a row's top mechanism. Ranking is what design choice uses; it must not wait until every absolute error is in budget.
2. Take the mechanism with the largest share of that row's critical path.
3. Look it up in the inventory:
   - status `assumed` or `absent`, share ≥ 3 % → **probe it** (this round);
   - status `measured`, share ≥ 3 % → the rule is wrong, not the number: write a probe that discriminates the current rule from an alternative;
   - share < 3 % → not allowed; go to the next mechanism on that row, then the next row.
4. If every valid row is within budget (median ≤ 10 %, none > 25 %, no inversion among design variants of one shape): take the inventory row that is `assumed` or `absent`, affects a test-set kernel, and is a search-space or hardware design variable; cheapest probe first.
5. If nothing qualifies: the loop is done for the current test set — write the final report and stop.

Hard limits that the scripts and the state file enforce:

- **Two experiments per open question**, counted per mechanism, not per round. `STATE.md` keeps the count; at 2 without a discriminating result the parameter is frozen at the best value, marked `unidentified`, and the question closed. Reopening needs a new row on the scorecard that points at it.
- **No row on the scorecard, no probe.** A mechanism that no valid row attributes to and that is not a design variable is not probed, however interesting.
- **A probe answers one question** with a sweep over which the hypotheses differ. A probe whose hypotheses predict the same numbers is not run.
- **Nothing below 3 % of a row's critical path is refined.** Half a cycle on a 16-cycle instruction is 3 %; it is refined only if a valid row shows it.

## 4. Fitting — allowed, bounded, visible

Residual fits are allowed because the model is never finished. The rules:

1. A fit attaches to one component or path (`launch`, `store tail per block`, `L2 return under load`), never to a kernel or a kernel family. Form: a constant, or a linear function of one named quantity.
2. Fits are computed only from probes and ablation records (the upstream 22-kernel GEMM set and its ablations count as development data). Never from the test set.
3. The parameter table marks it `fitted` with the data it came from. `STATE.md` keeps the fit ledger: name, component, value, data, the rows on whose critical path it sits and with what share.
4. The scorecard reports every row twice: **with fits** and **mechanisms only**. The gap between the two columns is the loop's own to-do list: the mechanism behind the largest gap is a candidate under §3 step 4.
5. A fit that improves absolute error without improving ranking (fewer inversions, higher correlation) is rejected: it is hiding a mechanism.
6. A fit on a search-space or hardware design variable is temporary by definition: it must be replaced by a mechanism before that variable is explored (what-if is refused on fitted parameters).

## 5. Scorecard — four numbers, not one

| Metric | Answers | Computed from |
| --- | --- | --- |
| absolute error: median and worst, with fits and mechanisms only | is the time right | every valid row |
| rank correlation (Spearman) and **inversion count** among design variants of one shape | can it choose designs | the variant groups of the test set |
| bottleneck agreement: the model's top critical-path unit vs the busiest counter pipe | is the bottleneck right | rows with counters |
| counts check pass rate (R1) | is the execution path right | every kernel row |

A round "improved the model" only if the targeted rows improve on absolute error **and** correlation/inversions do not worsen **and** no other row worsens beyond its tolerance. This is a record, not a veto. Acceptance is decided **per change**, with the scorecard run once per change switched off (ablation):

- A change backed by a probe (a measured parameter, or a rule a probe discriminated) **stays even if the scorecard worsens**. Two wrong mechanisms can cancel; fixing one exposes the other. Each row that worsens beyond tolerance becomes an open question "compensating mechanism missing", with the row and the change named; the selection rule picks it up like any other row.
- A fit, or a rule change without a probe behind it, is **rejected** if it lowers absolute error while adding inversions or lowering correlation (§4.5).

## 6. Round report — ten lines, appended to STATE.md

```
Round N: <batch, plain words: which mechanisms>
Question: <which rows, how far off, where the time went>
Hypotheses: A <…> / B <…> / C <…>
Result: <sweep summary; which hypothesis survived, or none>
Change: <what moved in the table / description / engine; new parameter with its source>
Scorecard: <targeted rows before → after; others unchanged within tol; Spearman x → y; inversions a → b>
New mechanism: <yes: inventory row n status absent → measured | no>
Fits: <added / removed / none; largest fit share on any critical path>
Next: <the mechanism §3 selects, one line>
Stopped: <only if the round did not complete: step and reason>
```

Write it so that someone who has not read the transcript understands what ran, what was learned, and whether the model got better. No other prose is needed.

## 7. When to stop and ask the user

Only these: an administrative action on spark (clock setting, driver, packages); a change to the test set, the budgets or the acceptance targets; a kernel class that needs a primitive the engine does not have (a schema change); a result that contradicts a `measured` parameter from an earlier round (two measurements disagree). Everything else is decided by the rules above and recorded.

## 8. Economy

Read `STATE.md` top, the scorecard top, the inventory row. Do not re-read the design documents each round. Do not re-derive the engine's structure: ask the what-if script. Do not write documents other than the report. One GPU batch per round; put every queued measurement (new test kernels, re-measurements) into that same batch. Keep the round's commits small and named `PERFMODEL: round N — <batch>`. Predictions and table rows are independent replays: run them in parallel (the table's process pool), never one by one.

**The test set grows every round.** A mechanism no test kernel exercises is never selected, so a narrow test set hides whole classes of mechanisms. Every batch adds at least one kernel from the 28 §5 table that is not on the scorecard yet (low-precision kernels included), until the table is covered; kernels the engine cannot express go on the scorecard as `unsupported: <missing primitive>`.

**Mechanism depth.** The inventory records for each measured mechanism whether its rule is *structural* (a resource, a port, a pipe, a queue: carries over to another design by changing the description) or *empirical* (a formula fitted over the probe's points, e.g. a latency of 24 + bytes/32: valid over the measured range only, re-measured on a new design). An empirical rule on a design variable is a candidate for a structural one under §3 step 4. A round that produces infrastructure and no new measured parameter is a failed round unless it is the first.
