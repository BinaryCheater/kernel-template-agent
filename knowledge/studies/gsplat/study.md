# gsplat NVIDIA source study

Status: extracted; pending independent review. Worker E fifth library. [Retained source](inputs/sources.json), Apache-2.0, CUDA C++ source reading; no GPU execution or timing.

Full-library representative-design coverage is reconciled in [inventory](inventory.md), with [concrete continuation](trace-additional-designs.md). Twelve candidate Template/ProofOutline pairs, including a local version2 extension with migration, are linked in [delta](delta/README.md). Existing rules used: output_tiling@2, physical_storage@1, reduction_partition@1, shared_memory_tiling@1. Cross-study comparisons pin exact versions in the trace.

Stop condition reached for this library: each materially distinct design has a concrete representative and disposition. Continue only to assigned NCCL; no commits.

Worker E queue closure: all six assigned libraries have representative-design dispositions. Final schema, retained-source, local-link and candidate-identity checks passed; see checks/queue-reconciliation.json. Stop after NCCL. No commits or GPU experiments were performed.
