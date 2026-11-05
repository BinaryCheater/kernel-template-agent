# shared_input_contraction_concatenation@1: shared input contraction concatenation

## When to consider

When several contractions consume the same complete input matrix and reduction axis, concatenate their weight output axes and compute one larger contraction, then expose the original output slices with their required casts and layouts.

## Computation before and after

When several contractions consume the same complete input matrix and reduction axis, concatenate their weight output axes and compute one larger contraction, then expose the original output slices with their required casts and layouts.

Given branches Y_j=X*W_j^T with equal X,K and compatible arithmetic, concatenate W_j along output features to Wcat. Compute Z=X*Wcat^T at the admitted common intermediate precision. The output column intervals partition Z and map back to the original Y_j. Expose each as a view only if its consumer accepts that stride; otherwise materialize the declared dense result, applying only its original dtype conversion/epilogue. A consumer may directly read its slice without saving a separate full matrix. Changing GEMM dimensions can select a different accumulation algorithm, so exact or error-bounded numerical admission is required independently of the real-arithmetic identity.

Recover the original computation: Restrict merged output columns to each original weight interval. Every reduction term then has the same input and weight coordinates as its separate contraction, recovering the branch output under the admitted numerical policy.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| slice_storage | compile / software | ["views", "dense_outputs"] | Consumers accept strided slices of the merged output, or each required slice is materialized densely. |

## Coupled constraints and scope

SGLang ops/moe/moe_front.py merges BF16 weights[896+3584,7168] over shared hidden input, uses FP32 Z, routes only its896-column gate slice and casts/materializes the3584-column latent slice to dense BF16. Source timings and bit-identity comments are not native validation here. Different inputs/K axes or unsupported mixed arithmetic are excluded.

Scope predicate: `(p.same_input_generation_reduction_axis_and_weight_dtype and p.weight_output_axes_disjointly_concatenated) and (p.merged_accumulation_and_output_precision_admitted_for_every_branch and p.branch_casts_occur_at_declared_stages) and (p.consumers_use_matching_slice_boundaries_and_layout and p.required_dense_outputs_materialized) and (p.merged_shape_supported_and_storage_fits and p.weight_concatenation_lifetime_valid)`.

- Precondition `pre_0` (implementation): `p.same_input_generation_reduction_axis_and_weight_dtype and p.weight_output_axes_disjointly_concatenated` — All branches use the same input/reduction coordinates; the concatenated weight ranges preserve every branch coordinate.
- Precondition `pre_1` (implementation): `p.merged_accumulation_and_output_precision_admitted_for_every_branch and p.branch_casts_occur_at_declared_stages` — A common merged intermediate satisfies the strictest consumer precision; later per-branch casts do not round a precision-sensitive branch early.
- Precondition `pre_2` (implementation): `p.consumers_use_matching_slice_boundaries_and_layout and p.required_dense_outputs_materialized` — Every downstream consumer sees the correct slice, stride and dense-layout requirement.
- Precondition `pre_3` (implementation): `p.merged_shape_supported_and_storage_fits and p.weight_concatenation_lifetime_valid` — The target supports the merged shape and its temporary; prepacked/concatenated weights retain matching generation.

## Local example and misuse

For a local instance of `contract`, identify the object `shared_input_contraction_concatenation` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/shared_input_contraction_concatenation@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/shared_input_contraction_concatenation@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Shared input with separate logical output regions and explicit temporary lifetimes.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `logical_equivalence` (proof): Restrict merged output columns to each original weight interval. Every reduction term then has the same input and weight coordinates as its separate contraction, recovering the branch output under the admitted numerical policy.
- `native_admission` (test): The selected native resource, arithmetic and publication recipe preserves all declared inputs/outputs.

Coordinate restriction proves the real-arithmetic identity because output feature intervals are disjoint and do not change K terms. It does not prove that a larger cuBLAS shape chooses the same reduction order. The MoE gate must retain FP32 through ranking; materializing all Z as BF16 first can change selected experts. Native GEMM-shape rounding remains the paired admission obligation; no GPU run is claimed.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
