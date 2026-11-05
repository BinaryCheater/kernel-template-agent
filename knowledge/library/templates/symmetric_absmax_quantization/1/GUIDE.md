# symmetric_absmax_quantization@1: zero centered absmax code group

## When to consider

When the requested low-precision representation is zero-centered with a shared group scale, reduce absmax and generate signed INT8 or E4M3 codes using that stored scale. Bind grouping, scale rounding and zero policy from the numerical contract; then choose ownership/fusion independently. Unsigned range-derived zero-point quantization is a different rule.

## Computation before and after

Per-group absmax then symmetric scale and payload conversion.

For each group reducea=max(abs(x)). With amax_floor replacea bymax(a,epsilon); with unit_scale use scale1 whena==0. Otherwise set raw_scale=a/max_code where max_code127 forint8 or448 fore4m3. Store FP32 raw_scale for fp32_ratio, or select the least admitted power of two not below raw_scale for ceil_pow2. Encode eachx using that same stored scale through the explicitly bound division/reciprocal, clipping and format conversion. Emit one scale and complete codes; code packing and metadata layout are separate.

Recover the original computation: Read code and shared scale as the requested symmetric quantizer result; decoded scale*code approximates original values under the specified quantization contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| group_size | compile / boundary | {"lo": 1, "hi": 65536} | Values sharing one numerical scale, fixed by representation; source FP8 group128 or entire INT8 row. |
| code_format | compile / boundary | ["int8", "e4m3"] | Signed INT8 range[-127,127] or NVIDIA E4M3 finite range[-448,448]. |
| scale_policy | compile / boundary | ["fp32_ratio", "ceil_pow2"] | Numerical contract chooses FP32 absmax/maxcode ratio or an upward power-of-two scale. |
| zero_policy | compile / boundary | ["amax_floor", "unit_scale"] | Floor the group maximum by caller epsilon before scale selection, or explicitly use scale1 for a zero group. |

## Coupled constraints and scope

Source signed symmetric INT8 or E4M3 quantizers with complete groups, positive finite scales and explicit epsilon/zero policy. Group128 FP8 AOT/TileLang and full-row INT8 ConvRot are concrete bindings; E8M0 minimum exponent, NVFP4 encoded E4M3 scales and arbitrary unsigned zero points are separate rules.

Scope predicate: `p.symmetric_quantizer_requested and p.finite_values and p.scales_positive_finite`.

- Precondition `pre_numeric` (semantic): `p.symmetric_quantizer_requested and p.finite_values and p.scale_reciprocal_and_conversion_recipe_explicit` — Scale/clipping/conversion order and INT8 nearest-even or E4M3 rounding are part of the requested numerical contract.
- Precondition `pre_scale` (implementation): `p.scales_positive_finite and p.ceil_pow2_domain_supported and p.epsilon_policy_matches` — Scale/reciprocal ranges and epsilon/zero behavior are supported by the chosen realization.
- Precondition `pre_owner` (semantic): `p.logical_groups_match_metadata and p.code_scale_generation_coherent` — One scale per logical group is published with exactly its corresponding code values.
- Constraint `c_group` (implementation): `p.length % group_size == 0 and p.complete_groups` — Every group consists of exactly the declared valid values.

## Local example and misuse

For a local instance of `map`, identify the object `zero_centered_absmax_code_group` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/symmetric_absmax_quantization@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/symmetric_absmax_quantization@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One FP32 scale per group plus signed integer or FP8 payload.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `group_max` (exhaustive): Maximum includes every logical group value once, plus only neutral padding if a future variant admits it.
- `scale_code` (test): Codes use the actual stored scale under the stated arithmetic and zero/epsilon policy.
- `range_rounding` (test): Scale and payload rounding/clipping implement the requested code format and supported range.

Multiplication by reciprocal differs from division; upward scale rounding changes payloads. This is a lossy numerical contract, not an exact storage permutation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
