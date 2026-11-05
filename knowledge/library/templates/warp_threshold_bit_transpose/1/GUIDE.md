# warp_threshold_bit_transpose@1: warp threshold bit transpose

## When to consider

When binary signs must be packed contiguously but input reads should remain coalesced, collect strided threshold bits per lane then transpose them into consecutive output bytes. Coordinate33 is lane1,bit1 in the input pack, and becomes output byte4,bit1. Coordinate7 becomes byte0,bit7. A dim34 tail leaves higher bits of byte4 zero; repeated all-one codes would hide a transpose error.

## Computation before and after

When binary signs must be packed contiguously but input reads should remain coalesced, collect strided threshold bits per lane then transpose them into consecutive output bytes.

Assign one warp to an input vector. For input tile base J, lane l collects bit b from x[J+l+32*b]>threshold[J+l+32*b], with invalid coordinates zero. Store that lane-local pack in warp scratch. Output-byte lane t reads bit floor(t/4) from scratch lanes (t%4)*8 through (t%4)*8+7, placing them into its output bits0..7. Store byte J/8+t only if within ceil(dim/8). This transposes the32-by8 bit tile while preserving the logical consecutive-bit output format.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| pack_bits | compile / software | [8] | Source pack_t uint8 byte; input tile is32*pack_bits scalar elements. |

## Coupled constraints and scope

NVIDIA binary quantization kernel uint8 pack. Source cross-lane shared-memory synchronization must be established for the target; the abstract mapping does not itself prove independent-thread scheduling safety. Threshold fitting is separate. Resource effects: Coalesced warp input loads, one byte scratch per lane and byte stores; transposition adds scratch/shuffle-equivalent work compared with scalar consecutive packing.

Scope predicate: `p.thresholds_feature_aligned and p.finite_comparison_policy and p.warp_scratch_ordered and p.full_warp_participation and p.output_stride_covers_tail`.

- Precondition `pre_0` (semantic): `p.thresholds_feature_aligned and p.finite_comparison_policy` — Per-feature threshold or zero and strict positive comparison match requested binary encoding.
- Precondition `pre_1` (semantic): `p.warp_scratch_ordered and p.full_warp_participation` — Lane scratch writes complete before cross-lane reads and scratch reuse follows all readers.
- Precondition `pre_2` (semantic): `p.output_stride_covers_tail` — Output allocation has ceil(logical_dim/pack_bits) bytes per row and tail bits begin zero.

## Local example and misuse

For a local instance of `map`, identify the object `warp_threshold_bit_transpose` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cuvs/delta/warp_threshold_bit_transpose@1/template.yaml`. Original outline: `knowledge/studies/cuvs/delta/warp_threshold_bit_transpose@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Coalesced warp input loads, one byte scratch per lane and byte stores; transposition adds scratch/shuffle-equivalent work compared with scalar consecutive packing.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `bit_mapping` (proof): Input coordinate J+l+32*b lands at output bit J+l+32*b exactly, with no swaps.
- `strict_threshold` (proof): Equality to threshold produces0 and threshold metadata belongs to the same input feature.
- `tail` (proof): Coordinates beyond dim do not set output bits or access input/threshold out of bounds.
- `scratch_lifetime` (proof): All participating lanes observe current tile scratch and finish before next overwrite.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
