# code_domain_group_requantization@1: code domain group requantization

## When to consider

When converting block-scaled FP4 into MXFP8, find maxima in the ordered magnitude-code domain before expanding every value. Pair the two16-value source groups, absorb the outer scale, and fold the new inverse power-of-two scale into each old group scale. This removes duplicated full decoding for scale selection while preserving code/group identity.

## Computation before and after

Requantize adjacent scaled low-float groups using code-domain maxima before expanding values.

Clear each E2M1 sign bit and reduce its ordered3-bit magnitude codes within each16-value input group, using packed max.u16x2 fields or an equivalent exact max. Decode only that maximum code, multiply by abs(input_group_scale*outer_row_scale), and max the adjacent two groups. Divide by448 and round upward to an encoded E8M0 power-of-two output scale. Multiply each original signed input scale by the inverse output power of two once per group, decode FP4 exactly through FP16 then promote to FP32, multiply by the folded scale and cast to E4M3. Emit one output scale per32 values. No separate NVFP4 or outer scale remains in the result. Zero groups produce zero codes under the stated zero-reciprocal policy.

Recover the original computation: Interpret output codes with their new E8M0 metadata as the requested MXFP8 quantization of the original decoded NVFP4 values including the outer row scale.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Pinned finite normal-scale NVFP4-to-MXFP8 conversion. Does not reuse a generic raw-product promotion rule or infer exactness for subnormal/extreme scales. Packed maxima require explicit logical groups, not reliance on inline-assembly pack order.

Scope predicate: `(p.e2m1_codes_and_direct_e4m3_group16_scales and p.outer_scale_immutable) and (p.output_mxfp8_group32_requested and p.groups_pair_without_crossing_rows) and (p.finite_scales_and_values and p.new_scale_normal_representable and p.power_two_fold_rounding_admitted)`.

- Precondition `pre0` (semantic): `p.e2m1_codes_and_direct_e4m3_group16_scales and p.outer_scale_immutable` — Input scale metadata describes the actual input codes and optional outer row scale is fixed.
- Precondition `pre1` (semantic): `p.output_mxfp8_group32_requested and p.groups_pair_without_crossing_rows` — Two adjacent16-value groups form each requested32-value group.
- Precondition `pre2` (semantic): `p.finite_scales_and_values and p.new_scale_normal_representable and p.power_two_fold_rounding_admitted` — The finite-range contract admits FP32 scale arithmetic and folding an inverse power-of-two scale; underflow/overflow is excluded.

## Local example and misuse

For a local instance of `map`, identify the object `code_domain_group_requantization` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/triton/delta/code_domain_group_requantization@1/template.yaml`. Original outline: `knowledge/studies/triton/delta/code_domain_group_requantization@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `code_maximum` (test): Ordered magnitude-code max equals max(abs(decoded E2M1)) in the same logical16-value group.
- `pair_grouping` (test): Adjacent maxima and per-value source scales are paired into exactly the same32-value output group.
- `scale_fold` (test): Folding the inverse power-of-two scale preserves the admitted FP32 rounding domain and native conversions.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
