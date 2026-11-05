# floor_anchor_group_quantization@1: floor anchored mx output groups

## When to consider

When an MX output consumer expects floor-anchored32-element groups, choose E4M3/anchor256 or E2M1/anchor4, combine the complete group maximum, floor its exponent and encode the resulting scale. This preserves that quantizer contract while allowing separate scale layout and reduction scheduling. At maximum500, MXFP8 selects scale1 rather than a ceiling recipe scale2; their codes and reconstructed values differ.

## Computation before and after

Select a group scale by flooring its maximum exponent relative to a fixed destination anchor.

For each group retain pre-global-scale FP32 values x. Form each local nonnegative absmax, multiply it by g in FP32 if global scaling is enabled, then combine all local maxima to m. Let b=8 for mxfp8 or2 for mxfp4. Strip the FP32 sign/mantissa of m, giving 2**floor(log2(m)); select s=2**(floor(log2(m))-b). On the admitted normal range, construct reciprocal r by exponent bits 0x7f000000-bits(s); it equals1/s exactly. Form multiplier h=RN_FP32(r*g) when global scaling is enabled, otherwise r; form z=RN_FP32(x*h), then RN-SATFINITE destination codes q. Store s as the matching UE8M0 exponent byte and q in the independently bound layout. A reader reconstructs s*decode(q), approximating the globally scaled output. The source zero-scale branch sets h=0 before optional global scaling, but its stored-scale conversion is outside this candidate.

Recover the original computation: Decode each group scale and destination code at the same semantic coordinate. Reconstructed s*decode(q) satisfies the admitted quantization/saturation contract for g*x; no equality to a ceiling-scale recipe is asserted.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| output_format | compile / software | ["mxfp8", "mxfp4"] | Destination contract: E4M3 with anchor256 or E2M1 with anchor4. The format fixes the anchor; it is not an independent tuning parameter. |

## Coupled constraints and scope

FlashInfer task-scheduled batched GEMM MX output on native supported targets/configurations. Source has per-pair and batched scratch phases, direct and TMA stores. Explicit E4M3 RN-SATFINITE packed TMA conversion is retained; default scalar conversion and E2M1 NVVM lowering require their conversion premise. Source zero/subnormal/infinite paths are recorded boundaries, not covered extensions. For a normal FP32 maximum of exponent k, normal selected scales require k>=-118 for MXFP8 or k>=-124 for MXFP4; finite maxima give upper k<=127.

Scope predicate: `p.group_size == 32 and p.groups >= 1 and (p.complete_group_coordinates_and_neutral_padding_explicit) and (p.finite_inputs_and_positive_finite_global_scale and p.fp32_max_scale_products_finite and p.code_multiplier_and_scaled_values_finite) and (p.selected_scales_positive_normal_fp32 and p.scale_exponents_between_minus126_and_126) and (p.rn_satfinite_code_cast_supported and p.fused_or_separate_fp32_multiply_order_explicit) and (p.ue8m0_decoder_and_layout_preserve_selected_scale_and_group) and (p.group_max_publication_and_scratch_reuse_ordered and p.code_and_scale_stores_publish_together) and (p.floor_anchor_rounding_and_saturation_error_admitted)`.

- Precondition `pre_groups` (implementation): `p.complete_group_coordinates_and_neutral_padding_explicit` — Every group contains32 semantic output coordinates on the declared axis; partial owners combine before scale selection, with neutral padding and identical scale/code ownership.
- Precondition `pre_values` (implementation): `p.finite_inputs_and_positive_finite_global_scale and p.fp32_max_scale_products_finite and p.code_multiplier_and_scaled_values_finite` — Group values are finite and the optional global multiplier g is positive finite (otherwise g=1); all FP32 max-times-g intermediates are finite. The composed code multiplier and scaled code inputs are finite too.
- Precondition `pre_range` (implementation): `p.selected_scales_positive_normal_fp32 and p.scale_exponents_between_minus126_and_126` — Each live group has nonzero maximum and its selected power-of-two scale and reciprocal are normal FP32. Zero, underflow and reserved metadata behavior require a separate extension.
- Precondition `pre_cast` (implementation): `p.rn_satfinite_code_cast_supported and p.fused_or_separate_fp32_multiply_order_explicit` — Bind RN finite-saturating destination conversion and source FP32 multiplication/FTZ behavior. Do not silently replace source multiplier composition by real-number division.
- Precondition `pre_metadata` (implementation): `p.ue8m0_decoder_and_layout_preserve_selected_scale_and_group` — UE8M0 exponent encoding decodes to the same selected positive normal scale at each group coordinate; physical packing/transpose remains separate.
- Precondition `pre_lifetime` (implementation): `p.group_max_publication_and_scratch_reuse_ordered and p.code_and_scale_stores_publish_together` — All partial maxima publish before readers; scratch is not reused until readers finish; consumer sees matching payload and scale generations.
- Precondition `pre_error` (implementation): `p.floor_anchor_rounding_and_saturation_error_admitted` — The numerical contract admits floor-anchor rounding and saturation, which may clip the maximum even though a larger power-of-two scale could avoid it.
- Constraint `c_groups` (semantic): `p.group_size == 32 and p.groups >= 1` — Complete32-element semantic groups; physical owners and output axis are separately bound.

## Local example and misuse

For a local instance of `contract`, identify the object `floor_anchored_mx_output_groups` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/floor_anchor_group_quantization@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/floor_anchor_group_quantization@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One UE8M0 scale per32 codes; physical layout and partial-max scratch are independently refined.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `floor_scale_identity` (exhaustive): Exponent truncation and anchor division produce the declared scale; exponent-only reciprocal is exact throughout the admitted normal range.
- `scale_code_identity` (test): UE8M0 decode returns the selected scale and never mixes group owners or generations.
- `complete_max` (test): The scale uses the maximum of all32 semantic elements, including pieces held by different warps; publication and scratch lifetime preserve it.
- `numeric_policy` (test): FP32 multiply order, rounding, clipping and reconstructed error satisfy the chosen destination contract.

The floor anchor can saturate values above destination range; zero/subnormal scale arithmetic and exponent-only reciprocals do not implement the normal-domain recipe. Global scale placement and output cast policy must remain explicit.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
