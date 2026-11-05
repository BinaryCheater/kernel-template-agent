# Attention reconstruction: experiment findings

9/9 outcomes recorded. 9/9 selected implementations passed hidden O and LSE checks; 2/9 met the full performance target.

Initial confirmation established 2/9 target outcomes. A separately recorded fixed three-window recovery measured all four initially unresolved sources, using their remaining original confirmation budgets. No model was reopened and no source was changed. The table below uses that recovery for all four eligible sources; initial measurements remain intact.

Both Template-assisted and direct implementation can reproduce the computation and reach the performance target. Success counts and costs do not establish an efficiency advantage from supplying Templates in this pilot.

A separate self-selection candidate repeatedly ran faster than the original. Two valid follow-up windows measured approximately 18%, 6% and 24% less time at S=512,2048,513. The third window failed the original-comparator stability check, so this promising result is retained but excluded from formal target counts. Its descending query order with interleaved heads schedules the most expensive causal tiles first; shape-specific tiling and storage reuse are also present in the frozen source.

No MIP solver or KTA performance-prediction model was invoked. Agents selected tile sizes, thread counts, staging and other implementation choices themselves, wrote concrete parameters into each candidate, and iterated using measured GPU correctness/timing feedback. This is a knowledge-conditioned coding and empirical tuning experiment; it does not evaluate KTA solver-driven parameter search or performance-model accuracy.

One attention operator was tested, with sequence lengths 512, 2048 and 513. Nine sessions mean three independent repetitions of each information condition. 320 is the number of library entries available for selection, not 320 kernel tasks.

The comparator is the unchanged, pinned FlashAttention-2 CUDA causal FP16 D128 source kernel, compiled for GB10. Q/K/V use BSHD [1,S,8,128]. Both implementations return FP16 O and FP32 natural-log LSE. Each uses its own complete GPU computation. The fixed target allows at most 11.1% more GPU time on every shape in all three accepted confirmation windows.

| Condition | Hidden checks passed | Stable three-window measurements | Full performance target |
| --- | --- | --- | --- |
| Supplied 18 Templates | 3/3 | 3/3 | 0/3 |
| Choose from 320 Templates | 3/3 | 2/3 | 1/3 |
| No Templates | 3/3 | 2/3 | 1/3 |

## Original time → generated time

Microseconds; parentheses show the change in elapsed GPU time. A positive percentage means the generated kernel takes longer. Values are medians of retained final-window medians. Rows marked unstable retain diagnostic numbers but do not establish a confirmed performance result.

| Condition / repetition | S=512 | S=2048 | S=513 | Measurement |
| --- | --- | --- | --- | --- |
| Supplied 18 Templates / 1 | 21.59 → 24.38 (+12.9%) | 147.56 → 161.57 (+9.5%) | 23.59 → 24.67 (+4.6%) | stable |
| Supplied 18 Templates / 2 | 21.68 → 22.54 (+4.0%) | 148.58 → 161.25 (+8.5%) | 23.69 → 26.46 (+11.7%) | stable |
| Supplied 18 Templates / 3 | 22.14 → 22.49 (+1.6%) | 148.81 → 159.00 (+6.8%) | 23.58 → 26.57 (+12.6%) | stable |
| Choose from 320 Templates / 1 | 22.20 → 22.54 (+1.5%) | 148.18 → 161.11 (+8.7%) | 23.62 → 26.64 (+12.8%) | stable |
| Choose from 320 Templates / 2 | 22.29 → 22.51 (+1.0%) | 148.26 → 159.83 (+7.8%) | 23.64 → 24.84 (+5.1%) | stable |
| Choose from 320 Templates / 3 | 22.25 → 18.16 (-18.4%) | 149.97 → 139.67 (-6.9%) | 24.49 → 18.49 (-24.5%) | unstable/unresolved |
| No Templates / 1 | 21.63 → 24.61 (+13.8%) | 147.59 → 189.64 (+28.5%) | 23.59 → 29.67 (+25.8%) | unstable/unresolved |
| No Templates / 2 | 21.66 → 22.38 (+3.3%) | 148.21 → 159.71 (+7.8%) | 23.61 → 26.65 (+12.9%) | stable |
| No Templates / 3 | 22.11 → 22.49 (+1.7%) | 148.15 → 157.95 (+6.6%) | 23.77 → 23.65 (-0.5%) | stable |

## Time, tokens and estimated cost

| Session | Active minutes | GPU waiting minutes | Input tokens | Cached subset | Output tokens | API-equivalent USD |
| --- | --- | --- | --- | --- | --- | --- |
| run-02 | 30.0 | 25.8 | 13,514,367 | 13,422,720 | 27,090 | 1.796 |
| run-04 | 22.3 | 25.0 | 6,614,882 | 6,539,904 | 11,168 | 0.916 |
| run-09 | 17.3 | 27.3 | 8,999,006 | 8,842,752 | 16,829 | 1.365 |
| run-01 | 21.0 | 31.1 | 10,522,982 | 10,368,384 | 15,978 | 1.506 |
| run-06 | 21.2 | 38.9 | 12,065,262 | 11,985,920 | 20,434 | 1.562 |
| run-08 | 26.0 | 28.9 | 14,200,136 | 14,113,792 | 19,217 | 1.776 |
| run-03 | 12.5 | 17.7 | 4,009,396 | 3,904,000 | 10,284 | 0.704 |
| run-05 | 13.6 | 14.1 | 2,964,337 | 2,876,032 | 10,282 | 0.567 |
| run-07 | 18.8 | 35.2 | 8,087,334 | 7,981,312 | 17,091 | 1.181 |

Total retained-price API-equivalent estimate: **$11.373**. This is a price-card estimate, not an account charge. Cached tokens are included in input tokens; repeated requests count repeated context. Reasoning is included in output. Resumed responses are deduplicated. Active time includes reading, thinking, compilation, transport and tests; verified GPU waiting is separate. These are the nine implementation-session costs, including interrupted work; one-time library preparation and coordinator work are separate.

Experiment elapsed time, including interruption and follow-up measurement: 81.4 minutes. This excludes prior preparation. The follow-up measurement has no model-token cost. All request-token totals reconcile with the retained native-session totals.

## Execution problems that affect interpretation

All nine sessions experienced a common early transport/global-stop interruption and resumed in the same native contexts with the original remaining budgets. GPU jobs were queued locally after the interruption. A monitor was then corrected to distinguish a proven child of the same container from an external GPU process. Raw rejected measurements were retained and their thresholds were not relaxed.

run-03 was prematurely closed after a coordinator code/receipt mismatch, with 17.5 active minutes still available. It is an infrastructure-truncated trial, not evidence that the model exhausted its ability or budget. Selected sources were frozen before hidden confirmation and were not repaired using final results. Accordingly this amended pilot cannot establish a clean causal advantage of Template information from group success counts.

Command review found run-08 copying its own development result records to recover lost feedback; those records also contain private development seeds. This is an exposure deviation. No hidden-confirmation inputs, other sessions, original target implementation or prior answer reads were found in the audited commands. Direct compile-only inspections used the shared locked recipe.

The supplied library has 312 canonical paired entries plus eight explicitly unpaired source proposals. The supplied-18 set and the selection pool use the same rule text. This tests written knowledge assistance, not a mechanically verified Template-application engine.

[Template preparation and duplicate handling](/Users/dingyi/Projects/agentic-codesign/kta/docs/runbooks/learndemo-1/template-consolidation-report.md).

[Frozen-source recovery protocol and raw measurements](/Users/dingyi/kta-data/dev/source-attention-control-v1/frozen-validation-v1).

[Full measurements, usage and evidence](/Users/dingyi/Projects/agentic-codesign/kta/docs/runbooks/learndemo-1/controlled-replay/results.md) · [Retained experiment records](/Users/dingyi/kta-data/dev/source-attention-control-v1)

[Figures, interpretation and reproduction](/Users/dingyi/Projects/agentic-codesign/kta/docs/runbooks/learndemo-1/controlled-replay/figures.md).
