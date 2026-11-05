# attention: guide-repair-prescribed-bounded (codex)

Correct on every prescribed shape. The per-shape performance aim is not met.

| Shape | Production median us (IQR) | Candidate median us (IQR) | Ratio |
|---|---:|---:|---:|
| 1 × 8 × 512 × 64 | 69.696 (1.936) | 58.352 (1.312) | 1.194 |
| 1 × 8 × 2048 × 128 | 160.176 (1.200) | 179.024 (0.928) | 0.895 |
| 1 × 8 × 514 × 64 | 21.920 (1.048) | 21.616 (6.120) | 1.014 |

Ratio is production latency / candidate latency. Raw samples remain in the run.

Settings: gpt-6.1-sol, effort high; exact native harness settings and frozen task/library are retained alongside the session.

Agent wall: 311.00 s; evaluation/transport within that wall: 74.13 s; GPU queue: 0.013 s; agent-job queue: 0.000 s.

Input including cache: 1096785; cached input: 1055744; uncached input: 41041; output: 10580. Monetary cost was not reported by the native harness. Original usage fields remain retained.

Evaluation batches: 6; failed/unfinished batches: ['003']. Final candidate matches measured batches: ['005'].

Compiler-only time is unavailable; first-call compilation/execution and validation are recorded separately from steady-state timing.

[Candidate](/Users/dingyi/kta-data/dev/template-replay/attention/guide-repair-prescribed-v2/workspace/candidate.py); [design note](/Users/dingyi/kta-data/dev/template-replay/attention/guide-repair-prescribed-v2/workspace/DESIGN.md). Retained run: `/Users/dingyi/kta-data/dev/template-replay/attention/guide-repair-prescribed-v2`. `attempts/` contains source snapshots, errors, environment records and raw timing samples; `session-state/` contains native transcripts, input, tool calls, settings and available accounting.

The changed physical-storage guide, template and proof were explicitly read twice. The final implementation uses dedicated Q/K/V/P shared buffers; the pipelined K/V alias failure was not repeated. This demonstrates bounded guide instantiation after explicit provision. The 0.894718 long-shape ratio still misses the per-shape aim; the agent’s aggregate claim does not change that result. This six-batch/900-second repair validation is separate from the original 20-batch comparisons.
