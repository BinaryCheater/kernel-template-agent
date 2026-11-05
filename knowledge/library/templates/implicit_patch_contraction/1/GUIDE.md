# implicit_patch_contraction@1: spatial patch matrix view

## When to consider

When a spatial cross-correlation needs repeated overlapping input windows, consider a virtual patch matrix so a GEMM path can consume those windows on load. Preserve channel-fast tap order and the leading-side padding in every coordinate, and choose patch_storage separately from the workload geometry. The pinned SM100 path uses im2col TMA rather than allocating the full matrix; stride/dilation/corners must fit its guarded descriptor fields, while grouped channels or nonzero padding need another binding or variant.

## Computation before and after

Virtual spatial patch matrix contracted with a channel/filter matrix.

For input A[b,d,h,w,c] and filter B[o,t,r,s,c], let the output spatial shape be (Z,P,Q), with each dimension floor((input+pad_low+pad_high- dilation*(filter-1)-1)/stride)+1. Flatten m=q+Q*(p+P*(z+Z*b)) and k=c+C*(s+S*(r+R*t)). Decode each (m,k) into those coordinates and define Patch[m,k]=A[b,z*stride_d-pad_low_d+t*dil_d, p*stride_h-pad_low_h+r*dil_h,q*stride_w-pad_low_w+s*dil_w,c], using zero when a spatial coordinate is outside the input. Define Weight[o,k]=B[o,t,r,s,c]. Materialize Patch or generate each requested tile on load. Contract Y[m,o]=sum_k Patch[m,k]*Weight[o,k], then map m back to (b,z,p,q). Bias, activation, quantization and physical transfer scheduling are separate.

Recover the original computation: Expand the matrix reduction into t/r/s/c and substitute the patch coordinates to recover the requested spatial cross-correlation without reversing filter taps.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| patch_storage | compile / software | ["materialize", "on_load"] | Whether to allocate the full patch matrix or form its tiles when loading; the pinned source selects on_load with im2col TMA. |

## Coupled constraints and scope

Three spatial axes, channel-last inputs and filters, independent batches, multiplicative contraction and zero padding. This is cross-correlation, not a filter reversal or transposed convolution. Pinned on-load binding is SM100 im2col TMA with source guards for signed corners[-16,15], tap offsets[0,31], stride1..8, tensor alignment and cluster geometry. The materialized branch is an algebraic alternative, not an executed source backend. Grouped channels, a different tap order, nonzero padding or another encoding needs an explicit variant or binding.

Scope predicate: `p.extents_positive and p.strides_positive and p.dilations_positive and p.padding_nonnegative and p.output_extents_match and p.input_filter_metadata_immutable and p.zero_fill_matches_contract and p.matrix_arithmetic_admitted and p.patch_mapping_supported and p.tail_accesses_safe and p.output_mapping_exact`.

- Precondition `pre_mapping` (semantic): `p.input_filter_metadata_immutable and p.zero_fill_matches_contract and p.matrix_arithmetic_admitted` — The input and filter mapping is stable; padding zero and matrix accumulation implement the requested numerical policy.
- Precondition `pre_backend` (implementation): `p.patch_mapping_supported and p.tail_accesses_safe and p.output_mapping_exact` — The selected patch implementation realizes the coordinate map with safe tails and exact output ownership.
- Constraint `c_geometry` (semantic): `p.extents_positive and p.strides_positive and p.dilations_positive and p.padding_nonnegative and p.output_extents_match` — Spatial output extents follow the cross-correlation geometry with positive strides and dilations.

## Local example and misuse

For a local instance of `contract`, identify the object `spatial_patch_matrix_view` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/implicit_patch_contraction@1/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/implicit_patch_contraction@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Full patch allocation or tile-local generation on load.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: materialized_patch
  level: gmem
  scope: device
  bytes: p.output_rows * p.reduction_elements * p.element_bytes
  when: patch_storage == 'materialize'
  note: Full logical patch allocation only; tiled on-load shared staging belongs to
    transfer rules.
overlap: []
```

## Obligations and numerical checks

- `coordinate_bijections` (exhaustive): Flattened output and reduction indices each invert their full logical coordinate domain exactly once.
- `patch_identity` (exhaustive): Every patch element selects the intended input coordinate or the padding identity with the correct leading-side padding.
- `contraction_identity` (test): The admitted matrix reduction and output map implement the same cross-correlation and numerical grouping contract.
- `implementation_bounds` (test): Descriptor fields or software coordinates encode every admitted parameter without truncation and all full/tail transfers are safe.

The coordinate identity is exact; matrix instruction precision and reduction regrouping must satisfy the caller's numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
