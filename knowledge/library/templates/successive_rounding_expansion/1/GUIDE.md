# successive_rounding_expansion@1: rounded residual components

## When to consider

When a tensor-core consumer accepts BF16 while weights are FP32, represent each finite safe-range weight by three successive BF16-rounded residual components. Preserve plane/coordinate identity and admit the consumer summation error separately. This trades three narrow contractions and6-byte storage for avoiding a direct FP32 operand path.

## Computation before and after

Three repeated rounding/residual extraction steps preserve an FP32 representation in BF16 components.

For each FP32 x, h=RN_BF16(x); r=fl_FP32(x-float(h)); m=RN_BF16(r); l=RN_BF16(fl_FP32(r-float(m))). Store h,m,l in three planes with identical coordinate order. Reconstruction adds their real decoded values. A consumer may contract each plane separately and add FP32 results only under a separate numerical policy; representation reconstruction does not certify that MMA rounding equals an FP32 GEMM.

Recover the original computation: Sum the three decoded arithmetic components per original coordinate. Under the stated rounding/range premise this recovers the original FP32 value; otherwise it defines an admitted approximation only after a new scope binding.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Pinned Triton FP32-to-three-BF16 encoding with contiguous input[N,K] and output[3N,K]. No generic arbitrary-component or TF32 claim. Storage costs6 bytes/value versus4 FP32; benefit requires a narrower arithmetic consumer and its own error admission.

Scope predicate: `p.fp32_input and p.bf16_rne_with_subnormals and p.finite_safe_expansion_domain`.

- Precondition `pre_domain` (implementation): `p.fp32_input and p.bf16_rne_with_subnormals and p.finite_safe_expansion_domain` — Source retrieval domain0 or2**-110<=abs(x)<2**128-2**119; narrower native flush/rounding behavior needs another binding.
- Precondition `pre_generation` (semantic): `p.components_share_source_generation and p.component_storage_complete` — Three component planes refer to identical coordinates and remain immutable through consumers.

## Local example and misuse

For a local instance of `store`, identify the object `rounded_residual_components` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/successive_rounding_expansion@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/successive_rounding_expansion@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Three BF16 planes; source plane offset=numel.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `reconstruction` (test): Three rounded residual components sum to the original finite FP32 input within the stated range and rounding model.
- `plane_identity` (exhaustive): Plane index and original row/column remain paired across encoding and consumer reshape.
- `native_rounding` (test): Native backend conversion/subtraction preserves the assumed rounding and subnormal behavior.

Representation identity is distinct from independently rounded component contractions. Native flush-to-zero, overflow, NaNs and Inf are outside scope.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
