# GEMM: prescribed Templates, original run

All three shapes are correct. The final unchanged candidate passed batches 18 and 19. The square shape misses the per-shape performance aim of 0.90; the narrow shape is near parity, and the tail shape is faster. Do not treat the agent's geometric-mean summary as satisfying the per-shape aim.

| Shape M × N × K | Production median us (IQR) | Candidate median us (IQR) | Production / candidate |
|---|---:|---:|---:|
| 1024 × 1024 × 1024 | 37.216 (0.512) | 42.416 (1.792) | 0.877 |
| 32 × 4096 × 4096 | 149.504 (4.312) | 150.784 (2.976) | 0.992 |
| 127 × 1000 × 769 | 23.168 (1.480) | 14.848 (1.488) | 1.560 |

Settings: Codex 0.166.0, GPT-6.1 Sol, high; TileLang 0.1.15 on GB10. Frozen [task](../replay-cases/gemm.md), library-v1, the same initial prompt used for all originals, warm repeated inputs, ten warmups and fifty event samples per shape. Production uses PyTorch/cuBLAS with reduced-precision reduction disabled. Fixed numerical tolerances were validated against FP64 before launch; evaluation inputs are refreshed privately per batch.

The session used 19 of 20 allowed batches in 640.28 s. Evaluation/transport occupied 184.87 s within that wall time; GPU lock wait totaled 0.040 s. Native agent-job queue wait was below 0.001 s. Input including cache: 2,299,168 tokens; cached: 2,254,208; uncached: 44,960; output: 22,951. Monetary cost was not reported by the native harness. Compiler-only time is unavailable; first-call compilation/execution is recorded separately from numerical checks and steady-state timing.

The agent read all six supplied guides and their structured rules/outlines. It selected output tiling, sequential reduction partition, compute instruction selection and physical storage; asynchronous transfer and buffer rotation served as readiness/lifetime guidance. Shape dispatch, warp policy, compiled-kernel caching and the compiler-managed pipeline came from its general GPU knowledge. It tried tile sizes, stages, warp policies and split-K. All nineteen candidates passed the bounded correctness suite. This run identifies a square-shape tuning gap, without evidence of a numerical or retrieval failure.

Best [candidate](/Users/dingyi/kta-data/dev/template-replay/gemm/prescribed-v1/workspace/candidate.py) and [design note](/Users/dingyi/kta-data/dev/template-replay/gemm/prescribed-v1/workspace/DESIGN.md). Complete retained run: `/Users/dingyi/kta-data/dev/template-replay/gemm/prescribed-v1/`; `attempts/` keeps sources, private inputs/settings, errors, environment snapshots and every timing sample; `session-state/` keeps the native transcript, settings, usage and archive. Coordinator result: `coordinator-result.json`. Source versions 18 and 19 match the final candidate exactly.

To rerun without modifying original evidence, copy manifest.json and preflight.json into a new run directory and call `scripts/template-replay remote-evaluate NEW_RUN FINAL_CANDIDATE --agent-active-s 0 --host spark-hk`. The wrapper creates a new remote attempt, preserves its outputs and holds the shared GPU locks. No model invocation is needed for this replay.
