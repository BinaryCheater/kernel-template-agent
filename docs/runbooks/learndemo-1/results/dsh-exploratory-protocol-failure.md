# DSH exploratory run: correct candidate, protocol deviation

Correct on every prescribed shape in the final official evaluator result; the per-shape aim is met. This interrupted exploratory run is excluded from the matched comparison because additional manual sweeps bypassed attempt accounting and did not retain raw timing samples. Sources, aggregate logs and the native session remain preserved. See `protocol-audit.json` in the retained run; the fresh protocol-repair run is separate.

| Shape | Production median us (IQR) | Candidate median us (IQR) | Ratio |
|---|---:|---:|---:|
| 1 × 8 × 512 × 64 | 19.664 (0.384) | 18.752 (0.184) | 1.049 |
| 1 × 8 × 2048 × 128 | 160.256 (0.584) | 151.104 (0.944) | 1.061 |
| 1 × 8 × 513 × 64 | 74.880 (2.456) | 61.248 (3.152) | 1.223 |

Ratio is production latency / candidate latency. Raw samples remain in the run.

Settings: deepseek-flash, effort high; exact native harness settings and frozen task/library are retained alongside the session.

Agent wall: 722.74 s; evaluation/transport within that wall: 30.22 s; GPU queue: 23.435 s; agent-job queue: 0.000 s.

Input including cache: 4293268; cached input: 4222080; uncached input: 71188; output: 68613. Monetary cost was not reported by the native harness. Original usage fields remain retained.

Evaluation batches: 3; failed/unfinished batches: ['001']. Final candidate matches measured batches: ['002', '003'].

Compiler-only time is unavailable; first-call compilation/execution and validation are recorded separately from steady-state timing.

[Candidate](/Users/dingyi/kta-data/dev/template-replay/attention/dsh-self-selection-v1/workspace/candidate.py); [design note](/Users/dingyi/kta-data/dev/template-replay/attention/dsh-self-selection-v1/workspace/DESIGN.md). Retained run: `/Users/dingyi/kta-data/dev/template-replay/attention/dsh-self-selection-v1`. `attempts/` contains source snapshots, errors, environment records and raw timing samples; `session-state/` contains native transcripts, input, tool calls, settings and available accounting.

Native accounting reconciliation: the final native assistant message was committed before interruption, while its stdout step-end event was absent. The authoritative totals above come from 48 unique native turn/step entries. Original stream-derived snapshots remain unchanged; `coordinator-summary-native-corrected.json` and `api-equivalent-estimate-corrected.json` contain the corrected analysis. Manual GPU queue time and raw samples outside the official evaluator were not recorded. Only the three official evaluator batches have complete sample/queue evidence.
