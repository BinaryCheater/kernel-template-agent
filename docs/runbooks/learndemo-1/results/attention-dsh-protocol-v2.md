# attention: self-selection-protocol-repair (deepseek_harness)

Correct on every prescribed shape. The per-shape performance aim is met.

| Shape | Production median us (IQR) | Candidate median us (IQR) | Ratio |
|---|---:|---:|---:|
| 1 × 8 × 512 × 64 | 19.664 (0.936) | 18.784 (0.352) | 1.047 |
| 1 × 8 × 2048 × 128 | 160.240 (1.648) | 145.504 (2.008) | 1.101 |
| 1 × 8 × 513 × 64 | 70.720 (1.560) | 60.864 (2.112) | 1.162 |

Ratio is production latency / candidate latency. Raw samples remain in the run.

Settings: deepseek-flash, effort high; exact native harness settings and frozen task/library are retained alongside the session.

Agent wall: 865.89 s; evaluation/transport within that wall: 158.17 s; GPU queue: 10.022 s; agent-job queue: 0.000 s.

Input including cache: 6629124; cached input: 6539520; uncached input: 89604; output: 56962. Monetary cost was not reported by the native harness. Original usage fields remain retained.

Evaluation batches: 14; failed/unfinished batches: ['001', '002', '004']. Final candidate matches measured batches: ['014'].

Compiler-only time is unavailable; first-call compilation/execution and validation are recorded separately from steady-state timing.

[Candidate](/Users/dingyi/kta-data/dev/template-replay/attention/dsh-self-selection-protocol-v2/workspace/candidate.py); [design note](/Users/dingyi/kta-data/dev/template-replay/attention/dsh-self-selection-protocol-v2/workspace/DESIGN.md). Retained run: `/Users/dingyi/kta-data/dev/template-replay/attention/dsh-self-selection-protocol-v2`. `attempts/` contains source snapshots, errors, environment records and raw timing samples; `session-state/` contains native transcripts, input, tool calls, settings and available accounting.

Conditions match the Codex attention self-selection task, initial prompt, library-v1, language, evaluator and budget. The public harness README adds an explicit evaluator-only instruction after the exploratory deviation; this clarification is disclosed. Native access audit found no separate target correctness/timing sweeps or answer-source reads. Native and stream usage totals agree across all 69 native assistant messages.

Timing variation remains visible: S=513 production/candidate medians were 22.048/20.944 us in batch 13 and 70.720/60.864 us in batch 14. These batches do not use identical final source bytes. Before/after environment snapshots do not establish clocks during that shape, so the agent’s clock explanation and strong stability wording are not adopted here.
