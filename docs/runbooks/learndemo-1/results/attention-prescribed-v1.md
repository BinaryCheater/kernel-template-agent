# attention: prescribed (codex)

Correct on every prescribed shape. The per-shape performance aim is met.

| Shape | Production median us (IQR) | Candidate median us (IQR) | Ratio |
|---|---:|---:|---:|
| 1 × 8 × 512 × 64 | 19.824 (1.296) | 18.912 (0.208) | 1.048 |
| 1 × 8 × 2048 × 128 | 159.600 (1.712) | 147.488 (1.944) | 1.082 |
| 1 × 8 × 513 × 64 | 70.528 (2.328) | 58.432 (1.744) | 1.207 |

Ratio is production latency / candidate latency. Raw samples remain in the run.

Settings: gpt-6.1-sol, effort high; exact native harness settings and frozen task/library are retained alongside the session.

Agent wall: 504.32 s; evaluation/transport within that wall: 175.31 s; GPU queue: 14.754 s; agent-job queue: 0.000 s.

Input including cache: 2034828; cached input: 1956224; uncached input: 78604; output: 16412. Monetary cost was not reported by the native harness. Original usage fields remain retained.

Evaluation batches: 15; failed/unfinished batches: ['004', '006']. Final candidate matches measured batches: ['015'].

Compiler-only time is unavailable; first-call compilation/execution and validation are recorded separately from steady-state timing.

[Candidate](/Users/dingyi/kta-data/dev/template-replay/attention/prescribed-v1/workspace/candidate.py); [design note](/Users/dingyi/kta-data/dev/template-replay/attention/prescribed-v1/workspace/DESIGN.md). Retained run: `/Users/dingyi/kta-data/dev/template-replay/attention/prescribed-v1`. `attempts/` contains source snapshots, errors, environment records and raw timing samples; `session-state/` contains native transcripts, input, tool calls, settings and available accounting.
