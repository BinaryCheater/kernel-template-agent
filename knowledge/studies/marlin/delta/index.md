# Marlin pool delta

No new Template, ProofOutline or Realisation is proposed. The pinned source reuses
`offline_operand_prepack@1`, `linearized_work_partition@1`,
`fused_dequantization@1`, `smem_partial_combine@1` and
`ordered_global_combine@1` with [source bindings](../trace.md). No existing identity
is replaced or admitted.

The [partial concrete Chain](chain.yaml) connects four of these mechanisms; shared
fragment pipeline, CTA tree and final-output layout remain explicit external edges.
It preserves reverse-K commit identity, FP16 partial storage and invocation lifetime.
A/B fragments, both scale layouts, output predicates and host M chunk/tail coverage
have retained source-equation checks. GPU synchronization, numerical acceptance and
compiler lowering remain open obligations of a future implementation/admission.
