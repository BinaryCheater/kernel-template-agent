# groupwise_low_float_quantization@1: low float group with encoded scale

## When to consider

When the requested output is a block-scaled low-float tensor, consider emitting each group's scale alongside its codes in the epilogue. Preserve the specified group/format pair and use the decoded stored scale for code generation, so rounding metadata cannot detach codes from their scale. The pinned path binds NVFP4 groups16 or MXFP8 groups32 with normalization1 and identity after quantization; finite-range, zero-group and reciprocal policies are explicit, while saturation/extreme underflow or another physical group map needs a verified extension.

## Computation before and after

Per-group absolute maximum, encoded shared scale, then low-float code conversion using that encoded scale.

For each group reduce amax=max(abs(v)). Let maximum_code be6 for nvfp4 or448 for mxfp8 and compute raw_scale=amax/maximum_code. Encode raw_scale as E4M3FN with nearest-even rounding for nvfp4, or as the least E8M0FNU power of two not smaller than raw_scale for mxfp8. Decode that same stored scale as s. Compute multiplier=min(admitted_reciprocal(s),FP32_max) and encode every v*multiplier to the output low-float format with nearest-even rounding. Emit code values and exactly one matching encoded scale per logical group. For zero groups, nvfp4 stores scale0 and clamps the positive infinite reciprocal to FP32_max; mxfp8 stores its minimum positive scale2**-127. Both produce zero code values. Output packing and physical metadata layout are separate refinements.

Recover the original computation: The pair of code values and encoded metadata is the result of the requested low-float quantizer, decoded as code_value*s; it does not recover arbitrary original real values exactly.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| format | compile / boundary | ["nvfp4", "mxfp8"] | Requested output representation; nvfp4 is E2M1 with E4M3FN scale and mxfp8 is E4M3FN with E8M0FNU scale. |
| group_size | compile / boundary | {"lo": 16, "hi": 32} | Contiguous values sharing one scale; source output binds16 for nvfp4 or32 for mxfp8. |

## Coupled constraints and scope

Requested low-float output groups with multiplicative scale and no zero point, normalization1, finite input values and the pinned16/32-element format pairs. Nonzero raw scales are within the metadata format and scaled values remain within finite code range; saturation, extreme underflow and NaN/Inf require a separately admitted extension. Zero groups have the explicit policies above. Source conversion helper requests E8M0 upward rounding and otherwise nearest-even; exact generated conversion/reciprocal behavior and per-thread logical grouping remain backend obligations. A new activation, normalization, format pair, physical group map or padding path needs a verified binding or variant.

Scope predicate: `p.low_float_quantizer_requested and p.finite_group_values and p.normalization_is_one and p.nonzero_scales_in_supported_range and p.scaled_values_within_finite_code_range and p.group_fragments_match_logical_groups and p.scale_metadata_ownership_exact and p.identity_after_quantization and p.conversions_match_format_policy and p.reciprocal_matches_admitted_policy and p.zero_group_policy_supported`.

- Precondition `pre_contract` (semantic): `p.low_float_quantizer_requested and p.finite_group_values and p.normalization_is_one` — The coarse operation requests this quantizer on finite pre-quantization values with source normalization1.
- Precondition `pre_range` (semantic): `p.nonzero_scales_in_supported_range and p.scaled_values_within_finite_code_range` — Nonzero scales fit the metadata format and normalized values fit finite output codes; overflow/underflow extension is separate.
- Precondition `pre_mapping` (semantic): `p.group_fragments_match_logical_groups and p.scale_metadata_ownership_exact and p.identity_after_quantization` — Each physical group matches its logical row/channels and emitted metadata; no later activation changes the represented values.
- Precondition `pre_backend` (implementation): `p.conversions_match_format_policy and p.reciprocal_matches_admitted_policy and p.zero_group_policy_supported` — The backend implements the specified scale/code rounding and admitted reciprocal behavior including zero groups.
- Constraint `c_pair` (implementation): `(format == 'nvfp4' and group_size == 16) or (format == 'mxfp8' and group_size == 32)` — This version supports only the two pinned code/scale/group format pairs.
- Constraint `c_groups` (semantic): `p.complete_groups or p.neutral_group_padding` — Any missing group elements are initialized to zero and do not affect the group maximum.

## Local example and misuse

For a local instance of `map`, identify the object `low_float_group_with_encoded_scale` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/groupwise_low_float_quantization@1/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/groupwise_low_float_quantization@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Low-float output codes plus one encoded scale per group.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `logical_group` (exhaustive): Each absolute maximum includes exactly one logical group plus only neutral initialized padding.
- `encoded_scale_used` (test): Output code generation uses the decoded stored scale rather than the unrounded raw scale.
- `format_policy` (test): Code/metadata conversion and zero-group reciprocal behavior implement the selected finite-format numerical policy.
- `metadata_identity` (exhaustive): Every visible code group emits exactly its own scale with no wrong-lane or padded-store ownership.

Quantization is intentionally lossy; approximate reciprocal, FP32 arithmetic and conversion boundaries need the caller's admitted error policy.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
