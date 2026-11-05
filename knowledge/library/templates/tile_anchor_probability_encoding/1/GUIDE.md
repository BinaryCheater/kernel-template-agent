# tile_anchor_probability_encoding@1: local anchor probability code scale factorization

## When to consider

When narrow P codes lose small weights in a running row frame, encode them relative to a tile-local score maximum and carry a shared block scale to PV. Sharing an anchor across K16 groups saves local maxima/scale choices but couples their precision. Preserve unrounded softmax mass separately and admit the resulting approximation, including scale underflow.

## Computation before and after

Factor probability weights through local score maxima before block-scaled low-bit PV.

For each row and anchor span, use scaled base2 logits x and a=max(valid x). With finite running reference r, form lambda=2**(a-r)/6 and z_j=6*2**(x_j-a), with masked z_j=0. Compute tile mass as lambda*sum(z_j) before code or scale conversion. Encode c_j=RN_sat_E2M1(z_j) and s=RN_sat_E4M3(lambda). Replicate s to each of anchor_span/16 K16 scale groups in that span. The PV consumer uses c_j*s, not z_j*lambda and not c_j*lambda. A fully masked span produces zero mass, codes and scale without subtracting two infinities. Correct the old mass and numerator by 2**(r_old-r) once, add the unrounded mass and decoded-code PV, and normalize by the accumulated unrounded mass. Scale rounding can erase or distort numerator weights without the same denominator change. Keep V decoding and final output cast explicit; no reciprocal of the rounded s is used to generate z.

Recover the original computation: Before casts, z_j*lambda=2**(x_j-r). After casts, recover the requested approximate PV operator using c_j*s and the separately preserved unrounded mass. This does not assert exact unquantized softmax.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| anchor_span | compile / software | {"lo": 16, "hi": 128} | Number of same-row keys sharing one anchor and E4M3 scale; source selects128. Other divisible spans are algebraic alternatives requiring their own backend binding. |

## Coupled constraints and scope

Pinned source MiniMax-H3 NVFP4 FP4-PV on SM100/103, key tile128 and one anchor per row/tile, repeated across eight K16 groups. Ideal span alternatives are inferred, not native source menu entries. Source uses lag8 online reference, FP32 mass and accumulators, E2M1 codes/E4M3 scales. Fully masked spans, scale rounding and FP16 partial-output checkpoints need their declared realization. No common gain or hierarchical reciprocal-scale quantizer is implied.

Scope predicate: `anchor_span % 16 == 0 and p.key_tile >= anchor_span and p.key_tile % anchor_span == 0 and (p.factorized_quantized_probability_operator_requested_and_error_admitted) and (p.complete_disjoint_key_tiles_and_exact_support_mask and p.valid_logits_finite_and_final_mass_positive) and (p.finite_shared_reference_and_old_mass_numerator_corrected_once) and (p.anchor_exponent_scale_cast_code_cast_and_accumulation_policy_explicit and p.scale_underflow_saturation_and_zero_anchor_policy_admitted) and (p.code_and_scale_coordinates_generation_and_consumer_match and p.scale_broadcast_retained_through_all_k16_consumers) and (p.selected_anchor_span_implemented_and_resources_fit and p.value_decoder_and_final_output_cast_separately_bound)`.

- Precondition `pre_contract` (semantic): `p.factorized_quantized_probability_operator_requested_and_error_admitted` — The selected approximation explicitly distinguishes unrounded softmax mass from block-scaled quantized PV weights.
- Precondition `pre_support` (semantic): `p.complete_disjoint_key_tiles_and_exact_support_mask and p.valid_logits_finite_and_final_mass_positive` — Every valid contribution appears once; masked lanes contribute zero and final mass is positive.
- Precondition `pre_frame` (semantic): `p.finite_shared_reference_and_old_mass_numerator_corrected_once` — The streaming reducer corrects both old numerator and old mass into the new frame before adding this tile.
- Precondition `pre_numeric` (implementation): `p.anchor_exponent_scale_cast_code_cast_and_accumulation_policy_explicit and p.scale_underflow_saturation_and_zero_anchor_policy_admitted` — Exp/casts/range policy is explicit; a scale rounded to zero may erase PV weight while retaining nonzero mass.
- Precondition `pre_pair` (implementation): `p.code_and_scale_coordinates_generation_and_consumer_match and p.scale_broadcast_retained_through_all_k16_consumers` — Every K16 group decodes the scale belonging to its anchor span, row and generation.
- Precondition `pre_backend` (implementation): `p.selected_anchor_span_implemented_and_resources_fit and p.value_decoder_and_final_output_cast_separately_bound` — Native realization supports the selected span; V quantization/layout and final casting are separate choices.
- Constraint `c_span` (implementation): `anchor_span % 16 == 0 and p.key_tile >= anchor_span and p.key_tile % anchor_span == 0` — Anchor spans partition the physical key tile into complete K16 scale groups; mask the logical tail.

## Local example and misuse

For a local instance of `map`, identify the object `local_anchor_probability_code_scale_factorization` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/tile_anchor_probability_encoding@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/tile_anchor_probability_encoding@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One logical E4M3 scale per row/anchor, replicated into the K16 operand metadata, plus E2M1 codes.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `factorization` (proof): Every unmasked z*lambda equals the intended weight in the common frame; empty spans have a defined zero branch.
- `mass_convention` (test): Denominator sums unrounded z*lambda, while PV consumes rounded c*s; these conventions are not silently interchanged.
- `scale_group_mapping` (test): Every code and K16 scale has the intended row/span, physical coordinate and publication generation.
- `range_and_error` (test): Scale underflow/saturation, exponent approximation, E2M1 conversion and output errors meet the admitted contract.

Even constant V can cease to return that constant because numerator and denominator quantize differently. Very small tile scales can round to zero; normalizing by reconstructed mass defines another operator.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
