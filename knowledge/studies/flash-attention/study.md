# FlashAttention NVIDIA design study

Assignment: full NVIDIA catalog pass by distinct computation, layout, pipeline,
quantization and scheduling design. Start with the catalog's CuTe DSL forward
SM90/SM100 path, then compare other forward/backward generations, sparse/MLA and
auxiliary CUDA extensions. ROCm backends are out of scope. Do not resume the
user-paused CUTLASS study or materialize that external dependency.

Pinned [source manifest](inputs/sources.json): full retained top-level archive and
tree. Root and CuTe subtree carry BSD-3-Clause licenses; individual auxiliary
headers may carry their own notices. CUTLASS/Composable Kernel/aiter submodules
are absent. [Starting pool](inputs/pool-snapshot.json) includes five tiny-cuda-nn
provisional pairs and earlier candidates, with historical versions retained.
Seed FA2/FA3 references are prior observations, not evidence for this revision.

Status: source-design pass complete. Fifteen provisional Template/ProofOutline
pairs cover the extracted choices; source-backed dispositions account for CuTe
forward/backward/combine, sparse MLA, CUDA FA2/FA3, legacy NVIDIA Triton attention
and auxiliary dense/normalization/rotary/loss/distributed groups. The final
[inventory](inventory.md) reconciles public/build/device families. The
[review](review.md) is the short retrieval menu and names remaining extraction
and native realization gaps.

Schema/pair/expression checks pass for all fifteen pairs. Retained logical
checks cover empty normalization, ring packing, backward checkpoints, sparse
index/frame/chunk mappings, dropout replay/sign encoding and auxiliary
retention/partitioned loss. These are source/logical evidence, not native GPU
execution, benchmarking or Store admission. No WP gate, commit or push belongs
to this assignment. CUTLASS remains user-paused.

Library stopping condition met: distinct designs have dispositions and remaining
gaps are named. The catalog goal remains active. Advance to FlashInfer using its
retained pinned revision; keep every source archive/tree used by this study.
