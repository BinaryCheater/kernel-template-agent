# hierarchical_low_float_scaling@1: nested decode scale

## When to consider

When a low-bit group scale has limited dynamic range, encode that group scale relative to a coarser amax-derived multiplier and keep the coarse inverse for decoding. Quantize data using the rounded stored group scale, not its ideal pre-rounding value. The next consumer must decode the same product of local and coarse scales.

## Computation before and after

Normalize local low-float scale codes by a coarse-domain multiplier.

For a coarse-domain maximum M>0 choose global encode multiplier S_enc=min((448*6)/M, FP32_max) and S_dec=1/S_enc under the pinned arithmetic. For local absmax a form stored E4M3 scale s=cast_E4M3(a*(S_enc/6)). Quantize each local input with E2M1 code q=cast_E2M1(x/(float(s)*S_dec)), preserving the declared conversion and rounding sequence. The decoder returns float(q)*float(s)*S_dec. A vector16 group uses sixteen same-row values; square16 shares one absmax over its 256 values. Row-domain and tensor-domain M may differ; each orientation of a bidirectional quantizer requires its own declared groups and matching scale metadata. Zero groups, alternative scale types and error-selected map4/map6 use additional policies outside this version.

Recover the original computation: Decode code q with the stored local scale and coarse inverse; this recovers the requested hierarchical quantized value, not the original unquantized real input.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| group_shape | compile / boundary | ["vector16", "square16"] | Requested per-row sixteen-value group or shared sixteen-by-sixteen block. |
| outer_domain | compile / boundary | ["tensor", "row"] | Logical domain of the coarse amax. Row-scaled source is restricted to vector groups and rowwise output. |

## Coupled constraints and scope

Complete vector16 or square16 NVFP4 groups, E2M1 codes and E4M3 local scales, coarse range factor 448*6. Source common/cast/nvfp4/core_nvfp4.cuh and quantize_transpose_nvfp4.cuh implement the paired two-level format. Zero/underflow scale policies, UE5M3 scales, reserved scale headroom 256, stochastic stream equivalence and map4/map6 error selection require distinct bindings. Native path is SM100-family gated; no execution is claimed.

Scope predicate: `p.cols % 16 == 0 and (group_shape != 'square16' or p.rows % 16 == 0) and (outer_domain != 'row' or (group_shape == 'vector16' and p.rowwise_output_only)) and p.nvfp4_e2m1_e4m3_recipe_requested and p.outer_amax_matches_domain and p.group_membership_exact and p.outer_amax_positive_finite and p.values_finite and p.stored_group_scales_positive_finite and p.scale_arithmetic_admitted and p.codes_scales_and_outer_inverse_paired and p.native_pack_and_consumer_match`.

- Precondition `pre_format` (semantic): `p.nvfp4_e2m1_e4m3_recipe_requested and p.outer_amax_matches_domain and p.group_membership_exact` — The requested E2M1/E4M3 representation, coarse maximum and local groups are bound to the same logical values.
- Precondition `pre_range` (implementation): `p.outer_amax_positive_finite and p.values_finite and p.stored_group_scales_positive_finite and p.scale_arithmetic_admitted` — This first version admits only positive finite coarse/local scales and admitted arithmetic.
- Precondition `pre_pair` (implementation): `p.codes_scales_and_outer_inverse_paired and p.native_pack_and_consumer_match` — Code packing, stored group scales and the coarse inverse have matching ownership and decoder semantics.
- Constraint `c_groups` (semantic): `p.cols % 16 == 0 and (group_shape != 'square16' or p.rows % 16 == 0)` — Pinned groups are complete.
- Constraint `c_row` (semantic): `outer_domain != 'row' or (group_shape == 'vector16' and p.rowwise_output_only)` — Pinned row-scaled variant has no square groups or columnwise output.

## Local example and misuse

For a local instance of `contract`, identify the object `nested_decode_scale` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/transformerengine/delta/hierarchical_low_float_scaling@1/template.yaml`. Original outline: `knowledge/studies/transformerengine/delta/hierarchical_low_float_scaling@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Packed E2M1 values, one E4M3 scale per local group and a coarse FP32 inverse.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `nested_decoder` (proof): The consumer uses q*s*S_dec and encoding used the same rounded s and S_dec.
- `groups` (test): Every code and scale belongs to the intended coarse domain and local vector/block.
- `native_rounding` (test): Scale casts, reciprocal and packed E2M1 conversion implement the requested numerical recipe.

Moving coarse normalization across the E4M3 scale cast or using an unrounded local scale changes the quantizer. Groupwise and tensorwise maxima are not interchangeable.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
