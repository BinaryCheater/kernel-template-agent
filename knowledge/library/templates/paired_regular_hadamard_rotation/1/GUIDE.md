# paired_regular_hadamard_rotation@1: paired orthogonal regular butterfly operands

## When to consider

Before an admitted quantized contraction, apply the same normalized regular-quartet rotation to activation and weightK groups. Choose64/128/256/512 elements per group; preserve identical coordinate maps and weight generation. Pure4-point products distribute a constant input uniformly, unlike Sylvester, while odd-bit groups include a final2-point factor. Benefits require numerical/performance evidence for the workload.

## Computation before and after

Paired normalized regular4-point butterfly rotation, with optional final2-point factor.

On each H-element block apply one regular4-point stage per base4 digit. For each quartet that differs in the selected two index bits, computeS=(x0+x1)+(x2+x3) and y_d=S-2*x_(d xor3). If log2(H) is odd, finish with a sum/difference2-point stage on the remaining top bit. Multiply every output by1/sqrt(H). Apply the same normalized transform to each activation and weightK block; contract transformed coordinates with the original output association. AnyINT8 quantizer and epilogue scales are separate refinements. Keep offline rotated weights immutable for every consuming activation generation.

Recover the original computation: Each regular4 matrix has orthogonal rows with norm2; the optional2-point factor has normsqrt2. The normalized tensor product is orthogonal, so paired operand inner products equal original inner products in exact arithmetic.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| group_size | compile / software | [64, 128, 256, 512] | Admitted rotation block along K; changing it changes quantization error and needs the numerical search contract. |

## Coupled constraints and scope

Source ConvRot block sizes64/128/256/512, BF16 source activation withFP32 regular butterfly and normalized paired weight transform. Different from Sylvester Hadamard. Mean is preserved across all output coordinates only for pure4-point products(H64,256); odd-bit sizes add a2-point stage. No measured quantization improvement claimed.

Scope predicate: `p.K % group_size == 0 and p.activation_weight_same_transform_and_group_origins and p.rotation_and_subsequent_quantization_error_admitted`.

- Precondition `pre_pair` (semantic): `p.activation_weight_same_transform_and_group_origins and p.rotated_weights_generation_matches` — Offline weight and online activation transforms use the identical matrix, coordinates and source weight generation.
- Precondition `pre_numeric` (implementation): `p.rotation_and_subsequent_quantization_error_admitted and p.transform_cast_recipe_fixed` — Exact dot preservation does not certify rounded transforms or subsequentINT8 quantization; casts, normalization and group choice are admitted.
- Constraint `c_shape` (implementation): `p.K % group_size == 0` — Both operands partition into the same complete reduction-axis rotation groups.

## Local example and misuse

For a local instance of `contract`, identify the object `paired_orthogonal_regular_butterfly_operands` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/paired_regular_hadamard_rotation@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/paired_regular_hadamard_rotation@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Offline rotated weight and online transformed activation; ownership/retention separate.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `orthogonal_pair` (proof): Both operands use identical orthogonal matrix/group origins, so every group dot is preserved before rounding.
- `index_mapping` (exhaustive): Register/lane quartet partners match the requested index digits and no group boundary is crossed.
- `numeric` (test): Normalization, transform casts and any following quantization meet the admitted contract for chosen group size.

Regular and Sylvester transforms are not interchangeable quantizers although both are orthogonal; rounding and group size change code scales and error.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
