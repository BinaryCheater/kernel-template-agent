# TileLang: source-design pass complete

| Kernel need | Proposed reusable choice | Main condition |
|---|---|---|
| Consume existing 2:4 zeros with sparse MMA | [Structured-zero compression](delta/structured_zero_compression@1/template.yaml) | Preserve every nonzero and pair values with native position metadata. |
| Reuse attention probabilities without retaining their matrix | [Probability recomputation](delta/softmax_probability_recomputation@1/template.yaml) | Match forward row statistics, operands, support and numerical policy. |
| Replace token-by-token linear state work with contractions | [Chunked decayed state](delta/chunked_decayed_outer_product_state@1/template.yaml) | Local causal products use incoming state; outgoing state advances in sequence order. |
| Expose matrix products inside a short triangular solve | [Blocked unit-lower inverse](delta/blocked_unit_lower_inverse@1/template.yaml) | Four 16-row blocks; diagonal solutions precede dependent off-diagonal products. |
| Select exact unsorted top-k without full sorting | [Radix threshold](delta/radix_threshold_selection@1/template.yaml) | Monotone keys, complete counts and an exact quota of threshold ties. |
| Scan a long top-k row with bounded candidate storage | [Streaming filter](delta/streaming_topk_filter@1/template.yaml) | Conservative threshold, exact compaction and space for an all-hit next tile. |

For example, a long sparse-attention indexer can scan score tiles, compact its
candidate set with byte-radix thresholds, then gather selected KV rows and run
online attention. Selection quality and support membership belong to the
operator; this chain does not establish equivalence to dense attention.

[Six Template/ProofOutline pairs](delta/index.md) passed
[schema, pair-reference and expression checks](inputs/schema-check.json).
Concrete source bindings and algebraic witnesses are in the
[GEMM scheduling](trace-gemm-scheduling.md),
[representation](trace-gemm-representations.md), [attention](trace-attention.md),
[recurrence/mixing](trace-recurrence-and-mixing.md) and
[selection/lowering](trace-selection-and-lowering.md) comparisons.

The [inventory](inventory.md) accounts for all NVIDIA example directories by
distinct design. Old rules explain most tiling, staging, warp roles, scale
transport and grouped work. Named gaps include balanced/final-owner Stream-K
realization, extra quantization/transform domains, prefix scan/reducer lifecycle,
WY/inter-chunk producers and native layout/launch/numerical bindings. No native
execution, benchmark, Store admission or end-to-end replay is claimed.

Advance to TransformerEngine. CUTLASS remains partially studied and user-paused.
