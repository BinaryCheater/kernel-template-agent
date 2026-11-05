# query_center_residual_scores@1: quantized query residual plus shared score correction

## When to consider

When a block of queries shares a large component before low-precision encoding, encode queries minus their shared mean and precompute mean times K once for the block. Broadcast that compact correction row and combine it with residual QK before softmax. Unlike common key translation, this correction varies across keys and cannot be discarded. Keep correction storage, head mapping, scale placement and seeded-MMA lifetime explicit.

## Computation before and after

Shared query mean/residual decomposition with compact key-dependent correction rows.

For each selected head or128-query block compute one mean mu and replace query q_i by residual r_i=q_i-mu before the chosen encoder. Compute C[g,j]=dot(mu_g,K_j) once per query group and key in FP32 storage, retaining the selected key operand and exact GQA head mapping. The score consumer broadcasts C[group(i),j] across all queries of its group and adds the decoded residual dot dot(r_i,K_j) before score scaling, bias/mask and softmax. It may seed the score accumulator with C and accumulate residual MMA into it, or add C after the residual contraction, provided the admitted arithmetic and lifetime contract holds. Do not omit C as if it were a row constant: it varies with key j. Do not scale only the residual or normalize the two score terms separately. Native residual quantization and finite subtraction change the approximation, so the ideal dot decomposition is not a bitwise equivalence claim. A separate key-centering step may remain in the supplied K; restoring this query term alone does not restore original-key-frame LSE. Recompute mean/correction for changing operand values and predicate padded output rows and key tails independently.

Recover the original computation: Expand dot(q_i-mu_g,K_j)+dot(mu_g,K_j) to dot(q_i,K_j) in exact arithmetic; the compact broadcast changes reuse, while admitted residual/correction rounding defines the selected approximation.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| center_scope | compile / software | ["head", "query_blocks"] | One query mean per head or one mean per128-query block; correction rows share the same scope. |

## Coupled constraints and scope

Forward affine attention scores with explicit query-center correction and an admitted residual quantizer. Source SM120/121 NVFP4 head64/128 uses one or128-query-block means, FP32 correction rows and a zero-stride shared broadcast before block-scaled QK. It independently centers K. Other mean granularities, raw-logit consumers, original-key LSE restoration, backward rules and nonlinear pre-correction transforms require separate bindings.

Scope predicate: `p.finite_query_key_center_and_correction_values and p.score_contract_is_dot_plus_declared_bias_mask and p.every_query_maps_to_exactly_one_center and p.correction_and_residual_use_same_key_queryhead_kvhead_map_and_generation and p.correction_added_once_before_mask_softmax_and_score_scale_applied_to_sum_once and p.mma_accumulation_preserves_seeded_correction and p.mean_subtraction_residual_quantization_correction_matmul_and_combined_error_admitted and p.correction_storage_range_and_precision_admitted and p.independent_key_center_and_returned_lse_frame_contract_explicit and p.correction_rows_and_scales_publish_before_matching_score_tile_and_retire_before_reuse and p.padded_query_and_key_validity_preserved and p.selected_center_scope_residual_encoder_and_correction_broadcast_implemented_and_resources_fit`.

- Precondition `pre_affine` (semantic): `p.finite_query_key_center_and_correction_values and p.score_contract_is_dot_plus_declared_bias_mask` — Finite operands and an affine dot-product score admit the decomposition.
- Precondition `pre_map` (semantic): `p.every_query_maps_to_exactly_one_center and p.correction_and_residual_use_same_key_queryhead_kvhead_map_and_generation` — Each row and GQA head use the same mean block, key operand and generation in both paths.
- Precondition `pre_merge` (semantic): `p.correction_added_once_before_mask_softmax_and_score_scale_applied_to_sum_once and p.mma_accumulation_preserves_seeded_correction` — Residual QK and correction sum before nonlinear operations; accumulator initialization cannot erase the correction.
- Precondition `pre_numeric` (implementation): `p.mean_subtraction_residual_quantization_correction_matmul_and_combined_error_admitted and p.correction_storage_range_and_precision_admitted` — Mean/subtraction casts, residual encoding, correction computation/storage and combined score error meet the policy.
- Precondition `pre_frame` (semantic): `p.independent_key_center_and_returned_lse_frame_contract_explicit` — This rule restores the query residual dot only; any separate key translation and original-frame LSE offset have their own binding.
- Precondition `pre_lifetime` (implementation): `p.correction_rows_and_scales_publish_before_matching_score_tile_and_retire_before_reuse and p.padded_query_and_key_validity_preserved` — Correction tiles survive all score readers and align with original valid/tail coordinates.
- Precondition `pre_backend` (implementation): `p.selected_center_scope_residual_encoder_and_correction_broadcast_implemented_and_resources_fit` — The selected center granularity, native encoded contraction and compact correction layout exist and fit.
- Constraint `c_shape` (semantic): `p.batch >= 1 and p.query_heads >= 1 and p.query_rows >= 1 and p.key_rows >= 1 and p.features >= 1` — Nonempty selected head/query/key domains.
- Constraint `c_rows` (implementation): `(center_scope == 'head' and p.correction_rows_per_head == 1) or (center_scope == 'query_blocks' and p.correction_rows_per_head == (p.query_rows + 127) // 128)` — Compact correction row count derives from the center scope, not a separate tuning choice.

## Local example and misuse

For a local instance of `contract`, identify the object `quantized_query_residual_plus_shared_score_correction` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/query_center_residual_scores@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/query_center_residual_scores@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: Compute mean times K once per head/block and reuse across its query rows while encoding smaller query residuals.. Storage: One FP32 score-correction row per head/query group, plus mean vectors; residual code/scales belong to the selected encoder.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: compact_score_correction
  level: dram
  scope: kernel
  bytes: 4 * p.batch * p.query_heads * p.correction_rows_per_head * p.key_rows
- id: query_centers
  level: dram
  scope: kernel
  bytes: p.center_element_bytes * p.batch * p.query_heads * p.correction_rows_per_head
    * p.features
overlap: []
```

## Obligations and numerical checks

- `score_identity` (proof): Residual and correction recover each original query/key dot before nonlinear normalization.
- `group_head_mapping` (test): Every correction row broadcasts to exactly its intended queries and GQA key head, including partial blocks and padding.
- `initialization_and_scale` (test): Correction is added once and not overwritten by MMA initialization; score scaling and masks apply to the completed sum.
- `numeric_and_lifetime` (test): Correction storage avoids unintended narrowing/overflow and matches residual/key generations through every consumer.

Quantized residuals plus a higher-precision correction generally differ from quantizing original queries. Narrow correction storage can overflow or lose the very shared component being recovered.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
