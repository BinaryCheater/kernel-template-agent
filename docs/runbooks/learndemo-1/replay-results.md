# Complete replay index

**Scope correction:** this indexes completed generic implementation runs. It does not establish reconstruction of a studied source-library kernel; see [the corrected claim](results/scope-correction.md). All original measurements and accounting remain retained.

All implementation sessions are terminal. The [one-page findings](results/summary.md) summarize the completed campaign. Every run retains native conversation, settings, source versions, failures, validation inputs, environment snapshots, queue time, raw timing samples and available accounting.

| Run | Ratios, shape order | Active min | GPU wait s | API token estimate |
|---|---|---:|---:|---:|
| [GEMM prescribed](results/gemm-prescribed-v1.md) | 0.877 / 0.992 / 1.560 | 10.67 | 0.04 | $0.545 |
| [GEMM self-selection](results/gemm-self-selection-v1.md) | 0.961 / 1.018 / 1.238 | 9.90 | 6.32 | $0.586 |
| [Attention prescribed](results/attention-prescribed-v1.md) | 1.048 / 1.082 / 1.207 | 8.16 | 14.75 | $0.517 |
| [Attention self-selection](results/attention-self-selection-v1.md) | 1.098 / 1.197 / 1.280 | 9.35 | 0.04 | $0.604 |
| [Attention categories](results/attention-category-disclosure-v1.md) | 1.108 / 1.166 / 1.216 | 9.59 | 99.89 | $0.650 |
| [Attention DSH](results/attention-dsh-protocol-v2.md) | 1.047 / 1.101 / 1.162 | 14.26 | 10.02 | $0.067–0.134 |

## Separate repair runs

| Run | Ratios, shape order | Active min | GPU wait s | API token estimate |
|---|---|---:|---:|---:|
| [Guide repair: self-selection](results/attention-guide-repair-v2.md) | 0.953 / 1.065 / 0.957 | 10.06 | 69.23 | $0.647 |
| [Guide repair: prescribed, six batches](results/attention-guide-repair-prescribed-v2.md) | 1.194 / 0.895 / 1.014 | 5.18 | 0.01 | $0.293 |

Both repair cases replace the original tail S513 with adjacent S514. The self-selection repair never read the changed physical-storage guide and repeated the alias trap; final correctness does not establish guide discovery. The six-batch/900-second prescribed validation read the guide and instantiated independent buffers. Its long shape misses the performance aim. The original comparisons keep library-v1; repairs use guide-only library-v2 with unchanged structured contracts/outlines.

[Exploratory DSH protocol deviation](results/dsh-exploratory-protocol-failure.md) is separate: final official candidate correct, but extra manual sweeps lack complete raw samples/accounting. Its interrupted native session remains resumable. Native assistant records reconcile the last usage absent from stdout; corrected analysis files are retained beside the originals.

## Token accounting

| Run | Input incl. cache | Cached input | Uncached input | Output |
|---|---:|---:|---:|---:|
| GEMM prescribed | 2,299,168 | 2,254,208 | 44,960 | 22,951 |
| GEMM self-selection | 2,680,112 | 2,631,424 | 48,688 | 22,515 |
| Attention prescribed | 2,034,828 | 1,956,224 | 78,604 | 16,412 |
| Attention self-selection | 3,233,890 | 3,179,904 | 53,986 | 17,850 |
| Attention categories | 3,229,178 | 3,166,464 | 62,714 | 20,825 |
| Attention DSH | 6,629,124 | 6,539,520 | 89,604 | 56,962 |
| Guide repair: self-selection | 2,954,550 | 2,895,744 | 58,806 | 24,003 |
| Guide repair: prescribed, six batches | 1,096,785 | 1,055,744 | 41,041 | 10,580 |

[Accounting and price sources](results/accounting-method.md); native account billing remains unavailable, with API-equivalent estimates explicitly separate. The additional exploratory DSH run used 4,293,268 native input tokens (4,222,080 cached) and 68,613 output tokens; its API-equivalent estimate is $0.0645–0.1290. Its 722.74-second wall is retained, but manual GPU wait/sample accounting is incomplete and excluded from the match.

Session wall and GPU waiting are separate; evaluator durations overlap session wall. Only accepted evaluator timing enters the tables. Compiler-only duration is unavailable; first-call compilation/execution remains retained.

[Kernel source and rerun index](results/implementations.md), [library consolidation](template-consolidation-report.md), [failure/guide analysis](results/failures-and-library-delta.md), [usage](replay-usage.md), [current state](STATE.md). Retained root: `/Users/dingyi/kta-data/dev/template-replay/`; machine index `final-results-index.json`. Source-study work was not restarted.
