# attention: self-selection (codex)

Correct on every prescribed shape. The per-shape performance aim is met.

| Shape | Production median us (IQR) | Candidate median us (IQR) | Ratio |
|---|---:|---:|---:|
| 1 × 8 × 512 × 64 | 19.712 (1.488) | 17.952 (0.592) | 1.098 |
| 1 × 8 × 2048 × 128 | 161.104 (0.280) | 134.576 (1.752) | 1.197 |
| 1 × 8 × 513 × 64 | 70.528 (3.912) | 55.120 (2.168) | 1.280 |

Ratio is production latency / candidate latency. Raw samples remain in the run.

Settings: gpt-6.1-sol, effort high; exact native harness settings and frozen task/library are retained alongside the session.

Agent wall: 561.12 s; evaluation/transport within that wall: 223.58 s; GPU queue: 0.039 s; agent-job queue: 0.000 s.

Input including cache: 3233890; cached input: 3179904; uncached input: 53986; output: 17850. Monetary cost was not reported by the native harness. Original usage fields remain retained.

Evaluation batches: 18; failed/unfinished batches: ['002', '009', '010', '011']. Final candidate matches measured batches: ['018'].

Compiler-only time is unavailable; first-call compilation/execution and validation are recorded separately from steady-state timing.

[Candidate](/Users/dingyi/kta-data/dev/template-replay/attention/self-selection-v1/workspace/candidate.py); [design note](/Users/dingyi/kta-data/dev/template-replay/attention/self-selection-v1/workspace/DESIGN.md). Retained run: `/Users/dingyi/kta-data/dev/template-replay/attention/self-selection-v1`. `attempts/` contains source snapshots, errors, environment records and raw timing samples; `session-state/` contains native transcripts, input, tool calls, settings and available accounting.
