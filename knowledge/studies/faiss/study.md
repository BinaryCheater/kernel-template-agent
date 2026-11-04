# Faiss NVIDIA source study

Status: extracted by distinct retained NVIDIA design; Worker E second library. Ownership and stopping condition follow ../assignments.md; no commits or GPU experiments.

Pinned source and license (MIT, root LICENSE): [manifest](inputs/sources.json). NVIDIA CUDA path, warp size 32; no specific SM or timings claimed. Scope: GPU index and primitive families registered by faiss/gpu/CMakeLists.txt, grouped by mechanism.

Pool: seed output_tiling@2, reduction_partition@1, multi_buffer_rotation@2; reviewed.yaml templates=[] at start. Read provisional streaming_topk_filter@1 (TileLang) and sorted_run_topk@1 (DeepGEMM). Their scratch append/compaction and deterministic head-merge contracts differ from collective-triggered lane queues; neither is silently widened. Stop after one representative-design pass and explicit gaps.

Progress: distinct retained NVIDIA design groups now concretely traced in trace-additional-designs.md. Candidate validation and final local reference/identity reconciliation passed.

Source-study closure: the concrete continuation supersedes initial unread dispositions. Local candidates validate, retained sources reopen, and their identities do not conflict with other study candidates at reconciliation. Device realization/proof and explicitly identified external dependency boundaries remain open; no major retained local design is closed from a directory listing alone.

Worker E queue closure: all six assigned libraries have representative-design dispositions. Final schema, retained-source, local-link and candidate-identity checks passed; see checks/queue-reconciliation.json. Stop after NCCL. No commits or GPU experiments were performed.
