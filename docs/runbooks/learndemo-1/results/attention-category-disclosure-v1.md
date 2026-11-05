# attention: category-disclosure (codex)

Correct on every prescribed shape. The per-shape performance aim is met.

| Shape | Production median us (IQR) | Candidate median us (IQR) | Ratio |
|---|---:|---:|---:|
| 1 × 8 × 512 × 64 | 19.792 (1.352) | 17.856 (0.288) | 1.108 |
| 1 × 8 × 2048 × 128 | 163.104 (1.544) | 139.824 (5.376) | 1.166 |
| 1 × 8 × 513 × 64 | 21.920 (0.800) | 18.032 (0.896) | 1.216 |

Ratio is production latency / candidate latency. Raw samples remain in the run.

Settings: gpt-6.1-sol, effort high; exact native harness settings and frozen task/library are retained alongside the session.

Agent wall: 675.04 s; evaluation/transport within that wall: 227.18 s; GPU queue: 99.893 s; agent-job queue: 0.000 s.

Input including cache: 3229178; cached input: 3166464; uncached input: 62714; output: 20825. Monetary cost was not reported by the native harness. Original usage fields remain retained.

Evaluation batches: 17; failed/unfinished batches: ['004', '010', '011']. Final candidate matches measured batches: ['017'].

Compiler-only time is unavailable; first-call compilation/execution and validation are recorded separately from steady-state timing.

[Candidate](/Users/dingyi/kta-data/dev/template-replay/attention/category-disclosure-v1/workspace/candidate.py); [design note](/Users/dingyi/kta-data/dev/template-replay/attention/category-disclosure-v1/workspace/DESIGN.md). Retained run: `/Users/dingyi/kta-data/dev/template-replay/attention/category-disclosure-v1`. `attempts/` contains source snapshots, errors, environment records and raw timing samples; `session-state/` contains native transcripts, input, tool calls, settings and available accounting.
