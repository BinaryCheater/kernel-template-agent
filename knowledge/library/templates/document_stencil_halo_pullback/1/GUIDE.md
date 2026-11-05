# document_stencil_halo_pullback@1: partitioned causal stencil forward and reverse halo

## When to consider

When a causal stencil spans rank fragments, route halo rows by absolute document tokens and send unrounded boundary preactivation gradients back to the actual input owners. Merge all tap contributions before the final cast; operands with identical row plans may share an exchange by packing channels.

## Computation before and after

Absolute-document history gather and transpose stencil adjoint routing, with per-operand or channel-packed exchanges.

Prepare row routes from immutable document/global-token coordinates. For fragment [a,b), forward halo k=0..2 names token a-3+k; find its actual owner, even across multiple short fragments, or use zero before document start. Gather original BF16 rows into destination-major sends, exchange and wait, then run the local width4 stencil and activation. For backward recompute FP32 preactivation gradients dpre for the first up to3 output rows of each fragment. For every predecessor fragment ending at e that needs output e+m, place that dpre in its unique send slot; the same output dpre may be copied to several predecessor consumers. Exchange FP32 rows and wait. Each input owner folds the next up to3 output dpre rows through the correct taps into its pending FP32 dx, alongside its local stencil pullback, before one BF16 output cast. Compute local tap gradients from this rank owned output positions with matching history. For channel_packed, concatenate operand columns on each identical row slot, multiply row splits by total channels, then unpack by channel prefixes and run unchanged per-operand kernels. Empty ranks preserve collective order. Retaining immutable forward history for checkpoint replay does not remove reverse dpre transport.

Recover the original computation: Expand each history and future-adjoint route to its absolute document token. Every forward stencil edge t<-t-lag has one original input value; its transpose edge returns dpre[t]*tap[3-lag] exactly once to that input owner. Copying one dpre to multiple predecessor fragments represents distinct input edges, not duplicate contributions to one dx. Channel packing preserves each operand edge identity.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| operand_exchange | compile / software | ["per_operand", "channel_packed"] | Use one forward/reverse halo exchange for each operand or concatenate channel columns of multiple operands sharing the same document-row plan into one exchange per direction. |

## Coupled constraints and scope

FlashInfer Cake deterministic KCP width4 depthwise causal convolution with SiLU on SM100/103/107; BF16 activations/dy/dx, FP32 taps and returned dpre, no bias and C divisible by8. Halo plan uses global document offsets and up to3 consumers per head row. KCP backward differentiates FP32 preactivation and therefore does not promise bitwise equality to the separate zero-history backend that differentiates a BF16 checkpoint. Tap gradients remain rank-local fixed-order partials. Generic network progress, arbitrary stencil widths and native arithmetic equivalence require their own binding.

Scope predicate: `(p.document_fragments_cover_each_token_once_and_keep_global_offsets) and (p.width_four_causal_taps_and_activation_gradient_stage_explicit) and (p.history_and_future_dpre_routes_resolve_absolute_document_tokens) and (p.send_receive_offsets_splits_and_replication_match_one_context_generation) and (p.raw_dpre_returned_and_owner_adds_each_legal_tap_once_before_cast) and (operand_exchange != 'channel_packed' or p.packed_operands_share_row_routes_and_channel_prefixes_exact) and (p.forward_history_and_reverse_dpre_publish_before_kernel_read_and_live_through_use) and (p.source_target_contiguous_shapes_and_halo_programs_supported)`.

- Precondition `pre_partition` (semantic): `p.document_fragments_cover_each_token_once_and_keep_global_offsets` — Nonempty fragments partition each document in chronological token coordinates, independent of rank/local ordering. Every halo token has one owning fragment; document boundaries denote zeros rather than tokens in adjacent packed documents.
- Precondition `pre_stencil` (semantic): `p.width_four_causal_taps_and_activation_gradient_stage_explicit` — The width4 depthwise stencil uses tap0 on t-3 through tap3 on t. Forward BF16 input/history and FP32 taps form the declared FP32 preactivation and SiLU recipe. Backward differentiates that FP32 preactivation, not a BF16-rounded checkpoint.
- Precondition `pre_routes` (implementation): `p.history_and_future_dpre_routes_resolve_absolute_document_tokens` — History row k for fragment start a resolves a-3+k; future dpre row m for its end b resolves b+m, for k,m=0..2. Missing before/after-document rows use sentinels. Short fragments may source three rows from several nonadjacent fragment/rank owners.
- Precondition `pre_slots` (implementation): `p.send_receive_offsets_splits_and_replication_match_one_context_generation` — Destination-major sends and source-major receives agree in row order and split counts, including same-rank peers. A head dpre row is replicated once per distinct predecessor-fragment consumer, up to3 consumers in this source. Slots have unique writers and no uninitialized sentinel read.
- Precondition `pre_backward` (implementation): `p.raw_dpre_returned_and_owner_adds_each_legal_tap_once_before_cast` — Backward returns FP32 dpre at boundary output positions, not already-rounded dx. Each input owner applies the appropriate FP32 tap to every future dpre contribution in its admitted local order, combines local contributions and performs one final BF16 dx cast. dweight is this rank fragment outputs partial, not a hidden global reduction.
- Precondition `pre_packing` (implementation): `operand_exchange != 'channel_packed' or p.packed_operands_share_row_routes_and_channel_prefixes_exact` — Packed operands share document partition, row routing and collective order. Concatenate operand channel columns with exact prefix widths, exchange once per direction and unpack contiguous views/copies without changing operand numeric stages. Different row plans cannot be co-packed by shape alone.
- Precondition `pre_lifetime` (implementation): `p.forward_history_and_reverse_dpre_publish_before_kernel_read_and_live_through_use` — Wait for forward history before local stencil reads and reverse dpre before input-gradient folds. Preserve original input/tap/context generations through recompute; a retained history handle removes forward exchange replay only, not reverse exchange.
- Precondition `pre_backend` (implementation): `p.source_target_contiguous_shapes_and_halo_programs_supported` — Concrete backend supports SM100/103/107, contiguous BF16 packed rows, channels multiple8, FP32 width4 taps and no bias. Empty ranks still participate in matching exchange and emit zero dweight; source graph/capture context readiness remains required.

## Local example and misuse

For a local instance of `contract`, identify the object `partitioned_causal_stencil_forward_and_reverse_halo` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/document_stencil_halo_pullback@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/document_stencil_halo_pullback@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Original BF16 history and FP32 dpre halo pools, replicated per distinct predecessor consumer; pending dx stays FP32 until final cast.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `global_token_routing` (test): Forward history and reverse future-output routes stay within the same document and find the actual owner across short fragments.
- `transpose_edges` (proof): Each stencil transpose edge contributes exactly once before final dx rounding, including replicated dpre consumers.
- `packed_exchange_identity` (test): Co-packed channels retain operand identity, row splits and numeric stages in both directions.
- `native_order_and_lifetime` (test): Activation derivative stage, FP32 accumulation order, waits, context generation and final BF16 cast match native requirements.

Immediate-predecessor-only history is wrong for short fragments. Returning rounded dx or changing SiLU preactivation precision changes gradients; channel co-packing is valid only for the same row plan.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
