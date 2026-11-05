# low_rank_scale_placement@1: raw dual scale or prepared low rank up factor

## When to consider

When many GEMMs reuse one low-rank adapter and base scale, pre-fold lambda/alpha into its up factor to share one final scale. Preserve raw independent scaling when factors change every call, alpha may be zero, or the added BF16 fold is outside the admitted error policy. Bind preparation generation and refold ordering explicitly.

## Computation before and after

Place low-rank adapter scale at runtime or pre-fold it relative to a common base-output scale.

Let R be the base residual matrix product, D the down result and U the BF16[N,rank] up factor. runtime_dual_scale computes Y=alpha*R+lambda*(D*U^T)+bias under the declared FP32 accumulation/scaling order, followed by the single output cast. prepared_up_fold requires alpha!=0: compute rho=RN_FP32(lambda/alpha), Uf=RN_BF16(FP32(U)*rho) once at preparation, retain its parameter snapshot, then compute Y=alpha*(R+D*Uf^T)+bias with the declared base-first correction order and final output cast. The extra BF16 fold defines a different admitted approximation. Refold Uf after every relevant U/lambda/alpha change; publication and exclusive buffer lifetimes apply to in-place graph refolds too. Lambda0 may use a backend-defined adapter-off branch. Alpha0 with a nonzero adapter is a raw dual-scale case, never a division-based fold.

Recover the original computation: Without factor rounding, distributivity gives alpha*(R+D*(lambda/alpha*U)^T)=alpha*R+lambda*D*U^T for nonzero alpha. With stored BF16 Uf, retain the explicit factor error and native accumulation policy; do not assert bitwise equality to raw scaling.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| placement | compile / software | ["runtime_dual_scale", "prepared_up_fold"] | Trade runtime independent scaling and pointer-only adapter switching for a prepared BF16 factor with common final alpha and explicit snapshot/rounding semantics. |

## Coupled constraints and scope

Source NVFP4 Cake prepared adapter on SM100/103 with ranks32/64/128; raw NVFP4/FP8 routes have separate target menus. SVDQuant shares the folded L1/alpha epilogue on SM100 and SM120-family backends, but full-chain preprocessing has its own numerical policy. FP8 prepared-adapter support is not implied. Snapshot object/version checks do not detect arbitrary external writes; explicit refold is required.

Scope predicate: `p.rows >= 1 and p.outputs >= 1 and p.rank >= 1 and (p.quantized_base_decoder_and_low_rank_components_explicit and p.base_down_up_coordinates_and_input_generation_match) and (p.alpha_lambda_finite_and_backend_scalar_range_admitted) and (placement != 'prepared_up_fold' or (p.alpha_nonzero_and_fold_ratio_finite and p.single_bf16_fold_rounding_error_admitted)) and (p.base_low_rank_accumulation_order_and_single_final_cast_explicit and p.bias_added_after_required_scaling) and (placement != 'prepared_up_fold' or (p.folded_up_alpha_lambda_same_prepared_generation and p.parameter_changes_refold_before_any_consumer)) and (p.preparation_completes_before_gemm_and_no_refold_overlaps_live_readers and p.selected_mode_rank_target_and_resources_supported) and (p.down_cast_base_quantization_correction_and_output_error_admitted)`.

- Precondition `pre_components` (semantic): `p.quantized_base_decoder_and_low_rank_components_explicit and p.base_down_up_coordinates_and_input_generation_match` — R is the requested decoded base product, D the declared down result and U its matching BF16 up factor; no quantized activation silently replaces the down input.
- Precondition `pre_scale` (implementation): `p.alpha_lambda_finite_and_backend_scalar_range_admitted` — Base alpha and adapter lambda have declared FP32/native range policy, including zero/subnormal behavior.
- Precondition `pre_fold` (implementation): `placement != 'prepared_up_fold' or (p.alpha_nonzero_and_fold_ratio_finite and p.single_bf16_fold_rounding_error_admitted)` — Prepared mode admits RN_FP32(lambda/alpha), one BF16 up-factor re-round and its effect on the low-rank correction; alpha0 stays on the raw path.
- Precondition `pre_order` (implementation): `p.base_low_rank_accumulation_order_and_single_final_cast_explicit and p.bias_added_after_required_scaling` — Base/correction accumulation and scaling order are bound; bias is not multiplied by the common base scale.
- Precondition `pre_snapshot` (implementation): `placement != 'prepared_up_fold' or (p.folded_up_alpha_lambda_same_prepared_generation and p.parameter_changes_refold_before_any_consumer)` — Folded U, alpha and lambda describe one immutable prepared snapshot; any changed weights/scalars refold before use, including writes outside framework version tracking.
- Precondition `pre_lifetime` (implementation): `p.preparation_completes_before_gemm_and_no_refold_overlaps_live_readers and p.selected_mode_rank_target_and_resources_supported` — Prepared writes publish before consumers, adapter switching waits for old readers, and backend/rank geometry is supported.
- Precondition `pre_numeric` (implementation): `p.down_cast_base_quantization_correction_and_output_error_admitted` — Down BF16 checkpoints, decoded base quantization and separate/fused corrections satisfy the numerical contract.
- Constraint `c_shape` (semantic): `p.rows >= 1 and p.outputs >= 1 and p.rank >= 1` — Nonempty base output and low-rank factor domains.

## Local example and misuse

For a local instance of `contract`, identify the object `raw_dual_scale_or_prepared_low_rank_up_factor` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/low_rank_scale_placement@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/low_rank_scale_placement@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Raw BF16 factor or retained folded BF16 factor plus parameter generation.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `scale_identity` (proof): The real nonzero-alpha identity and exactly-once common scaling preserve the intended low-rank contribution and unscaled bias.
- `fold_error` (test): FP32 ratio/product and one BF16 fold satisfy the admitted correction/output error, including tiny/zero scalars.
- `snapshot_coherence` (test): The prepared factor matches the current parameter snapshot and is never refolded while consumers read it.
- `input_channels` (test): The down path retains its declared original/smoothed activation and BF16 checkpoint rather than the residual quantizer output.

Folding a ratio into a BF16 factor re-rounds every element and changes correction error/association. A stale prepared factor can produce plausible but wrong outputs even with unchanged shapes/pointers.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
