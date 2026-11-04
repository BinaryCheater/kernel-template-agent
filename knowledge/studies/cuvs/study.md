# cuVS NVIDIA source study

Status: extracted by distinct NVIDIA design; Worker E third library. Full superproject retained in [manifest](inputs/sources.json); Apache-2.0 (root LICENSE). CUDA C++ and generated/JIT CUDA, no particular SM or measured performance claimed.

Question: identify reusable organization choices, especially CAGRA multi-CTA state ownership; inventory broader CUDA families without treating host registrations as executed kernels. Seed baseline: output_tiling@2, reduction_partition@1, physical_storage@1. Reviewed index templates=[] at start. Provisional Faiss collective_triggered_candidate_queue@1 is a different candidate-retention object; no dependency. Stop after this representative-design pass, retain unread families and continue assigned queue.

Current progress: seventeen local paired proposals; graph, quantized search, sparse/dense distance, covariance, stream batching and partition merge have concrete source traces. Local algebra witnesses and schema checks pass. Remaining before closure: balanced KMeans, quantizer training/error-bound producer, concrete dense tile policy and source registration reconciliation. See review.md for the current menu. Do not advance to another queue.

Closure: source and JIT/build inventory reconciled; twenty paired proposals plus concrete seed/provisional reuse, local witnesses and specific proof boundaries recorded. Continue to cuCollections under the same goal.

Worker E queue closure: all six assigned libraries have representative-design dispositions. Final schema, retained-source, local-link and candidate-identity checks passed; see checks/queue-reconciliation.json. Stop after NCCL. No commits or GPU experiments were performed.
