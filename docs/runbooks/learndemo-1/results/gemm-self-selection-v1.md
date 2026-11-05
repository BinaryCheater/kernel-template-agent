# gemm: self-selection (codex)

Correct on every prescribed shape. The per-shape performance aim is met.

| Shape | Production median us (IQR) | Candidate median us (IQR) | Ratio |
|---|---:|---:|---:|
| 1024 × 1024 × 1024 | 37.696 (1.664) | 39.232 (0.840) | 0.961 |
| 32 × 4096 × 4096 | 152.560 (6.208) | 149.824 (2.136) | 1.018 |
| 127 × 1000 × 769 | 23.232 (0.824) | 18.768 (0.872) | 1.238 |

Ratio is production latency / candidate latency. Raw samples remain in the run.

Settings: gpt-6.1-sol, effort high; exact native harness settings and frozen task/library are retained alongside the session.

Agent wall: 600.39 s; evaluation/transport within that wall: 194.63 s; GPU queue: 6.319 s; agent-job queue: 0.000 s.

Input including cache: 2680112; cached input: 2631424; uncached input: 48688; output: 22515. Monetary cost was not reported by the native harness. Original usage fields remain retained.

Evaluation batches: 20; failed/unfinished batches: none. Final candidate matches measured batches: ['018', '019', '020'].

Compiler-only time is unavailable; first-call compilation/execution and validation are recorded separately from steady-state timing.

[Candidate](/Users/dingyi/kta-data/dev/template-replay/gemm/self-selection-v1/workspace/candidate.py); [design note](/Users/dingyi/kta-data/dev/template-replay/gemm/self-selection-v1/workspace/DESIGN.md). Retained run: `/Users/dingyi/kta-data/dev/template-replay/gemm/self-selection-v1`. `attempts/` contains source snapshots, errors, environment records and raw timing samples; `session-state/` contains native transcripts, input, tool calls, settings and available accounting.
