# Source attention control: results

Status: template for future results; no numbers below are experimental observations.

State the answer first: whether reconstruction met the source-production target, and what additional benefit supplying Templates showed versus direct implementation. If execution/protocol evidence is incomplete, identify the missing result before presenting comparisons.

## Comparator and task

Name the original FlashAttention-2 source revision, causal FP16 D128 forward specialization, retained source/Template map and exact rerun command. State that its unchanged device kernel was compiled with a thin host binding and verified by an actual runtime trace. Identify original and generated computation as O plus natural-log LSE. State the three BSHD shapes, precision/tolerances, GPU and primary timing boundary. Explain any unsupported feature or recorded deviation in plain language.

## Every scheduled session

| Slot | Arm | Correct O and LSE on final hidden inputs | Meets production target on all shapes | Active time | GPU wait | Submissions | Outcome/deviation |
|---|---|---|---|---|---|---|---|
| 01 | Self-selection | pending | pending | pending | pending | pending | pending |
| 02 | Prescribed | pending | pending | pending | pending | pending | pending |
| 03 | No Template | pending | pending | pending | pending | pending | pending |
| 04 | Prescribed | pending | pending | pending | pending | pending | pending |
| 05 | No Template | pending | pending | pending | pending | pending | pending |
| 06 | Self-selection | pending | pending | pending | pending | pending | pending |
| 07 | No Template | pending | pending | pending | pending | pending | pending |
| 08 | Self-selection | pending | pending | pending | pending | pending | pending |
| 09 | Prescribed | pending | pending | pending | pending | pending | pending |

## Performance in ordinary units

For every correct final candidate, show each shape in a separate row. Never present a naked ratio without the underlying times.

| Session and shape | Original kernel time, microseconds | Generated kernel time, microseconds | Generated uses this much more/less time | Measurement spread | Target verdict |
|---|---|---|---|---|---|
| pending | pending | pending | pending | pending | pending |

The predeclared target permits up to approximately 11.1% longer time on every shape. Label a candidate faster only when the measured time is lower, and discuss variation. Retain all three confirmation windows and paired samples. Report eager-entry diagnostics separately from primary graph timings. An incorrect candidate has no accepted performance result.

## Benefit of Template information

Compare prescribed versus no-Template and self-selection versus no-Template, retaining all three independent runs per arm. Give the number that attained the target, each run's effort to the final confirmed source, final per-shape performance and variability. Show development first-correct/first-target observations separately as provisional milestones. For targets never reached, show not attained and the budget/censoring; do not average only the successful runs.

Report total input including cache, cached input, uncached input, output and available reasoning tokens; actual native dollar charge if returned, otherwise unavailable; API-equivalent estimate and rate source separately. Include Template reading/selection cost. Do not add evaluator time again to session wall or hide infrastructure waiting. Report resumed usage without double-counting cumulative totals.

Explain which rules were actually read/used, what the model supplied from its own knowledge and which original design decisions the library still fails to describe. State when no-Template runs perform equally or better. A lack of observed benefit on this kernel is a valid result. Do not claim causal utility from target attainment alone or from one best session.

## Evidence and next action

Link frozen campaign/public packages, source build/trace, prelaunch review, native sessions/settings, per-attempt sources/checks/timings, all protocol deviations, selected original-byte candidates and rerun commands. Keep old exploratory results separate. Choose a next step based on the observed difference or gap; do not automatically expand the benchmark/model matrix.
