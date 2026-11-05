# attention: guide-repair-self-selection (codex)

Correct on every prescribed shape; the per-shape performance aim is met. This self-selection run did not validate discovery of the guide repair: no observed tool command read physical_storage, and batch 10 repeated pipelined K/V alias failure. The final candidate uses separate buffers. The subsequent bounded prescribed-guide run tests the changed instructions directly.

| Shape | Production median us (IQR) | Candidate median us (IQR) | Ratio |
|---|---:|---:|---:|
| 1 × 8 × 512 × 64 | 19.904 (0.904) | 20.896 (0.432) | 0.953 |
| 1 × 8 × 2048 × 128 | 159.552 (1.752) | 149.808 (0.904) | 1.065 |
| 1 × 8 × 514 × 64 | 21.824 (0.336) | 22.816 (0.280) | 0.957 |

Ratio is production latency / candidate latency. Raw samples remain in the run.

Settings: gpt-6.1-sol, effort high; exact native harness settings and frozen task/library are retained alongside the session.

Agent wall: 673.10 s; evaluation/transport within that wall: 196.74 s; GPU queue: 69.232 s; agent-job queue: 0.000 s.

Input including cache: 2954550; cached input: 2895744; uncached input: 58806; output: 24003. Monetary cost was not reported by the native harness. Original usage fields remain retained.

Evaluation batches: 16; failed/unfinished batches: ['008', '010', '015']. Final candidate matches measured batches: ['014', '016'].

Compiler-only time is unavailable; first-call compilation/execution and validation are recorded separately from steady-state timing.

[Candidate](/Users/dingyi/kta-data/dev/template-replay/attention/guide-repair-v2/workspace/candidate.py); [design note](/Users/dingyi/kta-data/dev/template-replay/attention/guide-repair-v2/workspace/DESIGN.md). Retained run: `/Users/dingyi/kta-data/dev/template-replay/attention/guide-repair-v2`. `attempts/` contains source snapshots, errors, environment records and raw timing samples; `session-state/` contains native transcripts, input, tool calls, settings and available accounting.
