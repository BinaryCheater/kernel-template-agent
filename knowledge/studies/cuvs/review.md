# cuVS reusable design choices

Status: extracted by distinct NVIDIA design; pending independent review. Coverage is reconciled against build/JIT registrations in [inventory](inventory.md).

| Kernel need | Candidate | Useful choice |
|---|---|---|
| Several CTAs explore the same query | [Split-scope deduplication](delta/split_scope_traversal_deduplication@1/template.yaml) | Local candidate cache versus query-wide parent claim |
| Repeated resident query service | [Descriptor service](delta/persistent_descriptor_query_service@1/template.yaml) | Separate mailbox reuse from descriptor retirement |
| Group latency-sensitive requests | [Committed batches](delta/deadline_closed_committed_batch@1/template.yaml) | Timeout closes unused slots while preserving committed requests |
| Scan affine residual SQ codes | [Metric hoisting](delta/affine_residual_metric_hoisting@1/template.yaml) | Cache query/probe terms and preserve decoded-vector norm |
| Refine expensive distances | [Certified filter](delta/certified_distance_refinement_filter@1/template.yaml) | Bound versus complete-score/fixed-cutoff certificate; equality policy explicit |
| Fuse distance and nearest result | [Pair combine](delta/tile_argmin_pair_combine@1/template.yaml) | Publish a consistent score/ID pair under contention |
| Merge shards or appended tiers | [Partition merge](delta/partition_topk_candidate_merge@1/template.yaml) | Global IDs, partial-batch stride and root/tree completion |
| Incrementally improve a graph | [Local joins](delta/incremental_neighbor_local_join@1/template.yaml) | New-new/new-old sampled pair work and exclusive graph update |
| Shrink a route graph | [Occlusion pruning](delta/occlusion_bounded_neighbor_pruning@1/template.yaml), [detour pruning](delta/ranked_detour_neighbor_pruning@1/template.yaml) | Geometric ratio versus ranked two-hop alternatives; distinct heuristics |
| Quantize with explicit error policy | [Anisotropic update](delta/anisotropic_residual_coordinate_update@1/template.yaml), [round-away conversion](delta/saturating_affine_round_away_quantization@1/template.yaml) | Ordered residual projection updates versus scalar endpoint format |
| Compact threshold bits / integer dots | [Bit transpose](delta/warp_threshold_bit_transpose@1/template.yaml), [code dot](delta/bitplane_integer_dot_reconstruction@1/template.yaml) | Preserve logical coordinates and unsigned integer values |
| Sparse pairwise metric | [Cached segmented join](delta/cached_sparse_row_segmented_join@1/template.yaml) | Dense/hash row cache and segmented atomic combine |
| Weighted full covariance | [Centered Gram](delta/weighted_centered_gram_covariance@1/template.yaml) | Fixed mean generation, batch accumulation and explicit denominator |
| Stream host batches into GPU work | [Event staging](delta/event_ordered_host_device_staging@1/template.yaml) | Separate host-copy and device-compute lifetimes |

Concrete composition: composite/index.cu searches child indexes on stream-pool streams, offsets each child's valid IDs by preceding child sizes, packs each query's child winners, waits for all child streams, then selects global k. Two children of sizes100/200 map local ID0 to global0/100. With3 queries and k2, the packed row has4 candidates and stride4. This instantiates partition_topk_candidate_merge@1 relative to the candidates actually supplied by the approximate child searches.

[Concrete source decisions](trace-additional-designs.md) are the evidence entry point. Each new candidate has a neighboring proof.yaml with a mechanism-specific invariant and finite-work rank. The [pinned pool](inputs/pool-snapshot.json) records actual seed versions, including multi_buffer_rotation@2. Those CTA barrier semantics are not reused as proof of CUDA stream events.

Checks: `.venv/bin/python knowledge/studies/cuvs/checks/replay.py` passed the local anisotropic delta, asymmetric bit-plane dot, transpose coordinates/tails and signed-bound counterexample; [observed result](checks/replay-result.json). Twenty current pairs pass Template/ProofOutline schemas. Device ordering, exact FP32 boundary rounding, quantizer bound certificates and recall are separate open obligations. No GPU execution or performance measurement.

Closure: balanced/hierarchical KMeans, quantizer producer/error metadata and concrete CUDA/cuTile tile bindings are now traced. Additional choices: [donor reseed](delta/donor_seeded_centroid_reinitialization@1/template.yaml), [hierarchical center budget](delta/hierarchical_centroid_budget_seeding@1/template.yaml), [row-constant rank score](delta/row_constant_rank_score@1/template.yaml). The RaBitQ error coefficient supplies a heuristic envelope, so no deterministic certificate is claimed. The remaining obligations above are recorded for independent review and realization checks.
