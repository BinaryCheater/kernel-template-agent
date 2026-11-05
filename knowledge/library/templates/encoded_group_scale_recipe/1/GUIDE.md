# encoded_group_scale_recipe@1: encoded group scale recipe

## When to consider

When a block-scaled output needs a different format or scale-rounding recipe, select both explicitly and use the decoded stored scale to form codes. Upward and downward exponent recipes use different divisors; substituting one for the other changes the quantizer. Keep group identity, packed-code order and zero-group behavior visible.

## Computation before and after

A low-float group selects a scale-format and rounding recipe and quantizes using that exact encoded scale.

Reduce max(abs(valid values)) in each logical group, excluding invalid positions. UE8M0 up divides by the code finite maximum (6,448,57344) and rounds scale up to a power of two by exponent/mantissa bits. UE8M0 down divides by the maximum code power of two (4,256,32768) and rounds scale down to a power of two. Direct E4M3 divides by the finite maximum and casts nearest-even; up additionally increments a positive encoding if it undershoots, saturating scale encoding at448. Use the rounded FP32 scale represented by that encoding for reciprocal and value conversion. Zero groups use zero reciprocal and emit zero codes; UE8M0 metadata byte0 decodes to2**-127. Invalid codes become zero. FP4 values use satfinite nearest-even code conversion and even-index nibble low, odd high. Decode with the same metadata; do not claim lossless recovery of original values.

Recover the original computation: The emitted code/metadata pair is the requested rounded quantizer result; inverse decoding recovers represented values, not arbitrary original inputs.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| code_format | compile / software | ["e2m1", "e4m3", "e5m2"] | E2M1 codes pack two per byte. |
| scale_format | compile / software | ["ue8m0", "e4m3"] | Power-of-two unsigned exponent or direct floating scale. |
| rounding | compile / software | ["up", "down", "nearest"] | Scale rounding; not value-code rounding. |
| group_size | compile / software | {"lo": 1, "hi": 65536} | 32 for OCP MX,16 for NVFP4. |

## Coupled constraints and scope

Finite groups within supported normal scale ranges. E4M3 scale groups16 bind NVFP4; UE8M0 groups32 bind MXFP4/MXFP8. This adds MXFP4/E5M2 and explicit down/up scale variants absent from groupwise_low_float_quantization@1; exceptional groups and subnormal scale construction remain outside the candidate domain.

Scope predicate: `((scale_format == 'e4m3' and code_format == 'e2m1' and group_size == 16 and rounding in ['up','nearest']) or (scale_format == 'ue8m0' and group_size == 32 and rounding in ['up','down'])) and (p.requested_recipe and p.finite_valid_values and p.groups_have_valid_elements) and (p.nonzero_scales_normal_and_representable and p.no_scale_encoding_overflow) and (p.format_rounding_pair_supported and p.cast_reciprocal_and_saturation_admitted)`.

- Precondition `pre0` (semantic): `(scale_format == 'e4m3' and code_format == 'e2m1' and group_size == 16 and rounding in ['up','nearest']) or (scale_format == 'ue8m0' and group_size == 32 and rounding in ['up','down'])` — Pinned code/scale/group/rounding pairs are coupled, not a Cartesian product.
- Precondition `pre1` (semantic): `p.requested_recipe and p.finite_valid_values and p.groups_have_valid_elements` — The caller requests this lossy quantizer and invalid lanes are excluded from amax.
- Precondition `pre2` (semantic): `p.nonzero_scales_normal_and_representable and p.no_scale_encoding_overflow` — Nonzero group scales fit the declared normal metadata domain; extreme underflow/overflow is outside this binding.
- Precondition `pre3` (semantic): `p.format_rounding_pair_supported and p.cast_reciprocal_and_saturation_admitted` — UE8M0 supports up/down, direct E4M3 supports up/nearest. Code rounding/saturation is an explicit numeric contract.

## Local example and misuse

For a local instance of `map`, identify the object `encoded_group_scale_recipe` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/triton/delta/encoded_group_scale_recipe@1/template.yaml`. Original outline: `knowledge/studies/triton/delta/encoded_group_scale_recipe@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `scale_recipe` (test): Selected divisor, scale rounding and zero-group policy match the requested format recipe.
- `code_scale_identity` (proof): Each code was computed with the rounded stored scale belonging to its own group and logical coordinate.
- `native_conversion` (test): Hardware/software FP4 and FP8 conversion, reciprocal and saturation implement the admitted numeric policy.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
