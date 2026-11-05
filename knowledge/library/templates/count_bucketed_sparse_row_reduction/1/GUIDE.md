# count_bucketed_sparse_row_reduction@1: count bucketed sparse row reduction

## When to consider

When rows have a small bounded contributor axis and varying sparsity, bucket by active count and enumerate only bitmap members. Keep original row IDs through the permutation so dense output order is preserved. Static short loops and heavy row subtiles are independent scheduling choices; feature-dependent masks cannot share one bitmap.

## Computation before and after

Rows are stably bucketed by active contributor count, then reduced by ascending bitmap member indices.

Count nonzero contributor-mask entries per row. Within each planner CTA interleaved row enumeration, stable-partition into counts>=4, counts3, counts2, counts<=1, keeping the original row IDs and scattering back to that same enumeration of destination positions. This is not a global stable sort. For a tile of reordered rows dispatch from its maximum count: at most1, exactly2, exactly3 use those static loop bounds; mixed/heavy tiles use the full reduction extent and may split independent row subtiles. Build each row uint32 bitmap from nonzero mask entries. At every iteration obtain ffs(bitmap)-1, mask an empty bitmap, then clear its least significant set bit. Load only that contributor, apply its micro/global/per-element scales in declared order, and add. An empty row produces zero. Chain adjacent feature tiles under the same row metadata, then store at original row IDs. This is count-class ordering, not least-loaded heap assignment.

Recover the original computation: Invert the row permutation and expand the bitmap loop into the original masked contributor sum in ascending contributor index order.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| chain_factor | compile / software | {"lo": 1, "hi": 65536} | Adjacent feature tiles processed by one row executor. |
| heavy_subtiles | compile / software | {"lo": 1, "hi": 65536} | Independent row subtiles for the heavy bucket (source binds2). |

## Coupled constraints and scope

Feature-broadcast sparse masks with at most32 contributors; general per-feature masks use the separate ordinary masked loop. Reordering is safe for independent rows. Source thresholds1/2/3 are a compiled menu, not a measured universal optimum.

Scope predicate: `(p.mask_feature_broadcast and p.reduction_extent <= 32 and p.nonzero_mask_means_active) and (p.row_permutation_exact and p.original_row_identity_retained and p.bucket_limits_cover_counts) and (p.admitted_add_order_and_decode_scale_policy and p.output_tiles_disjoint)`.

- Precondition `pre0` (semantic): `p.mask_feature_broadcast and p.reduction_extent <= 32 and p.nonzero_mask_means_active` — One contributor bitmap describes every feature of a row.
- Precondition `pre1` (semantic): `p.row_permutation_exact and p.original_row_identity_retained and p.bucket_limits_cover_counts` — Reordering changes execution order only; dispatch limits cover every active contributor.
- Precondition `pre2` (semantic): `p.admitted_add_order_and_decode_scale_policy and p.output_tiles_disjoint` — Ascending contributor accumulation, scale decoding and output ownership match the requested numeric contract.

## Local example and misuse

For a local instance of `reduce`, identify the object `count_bucketed_sparse_row_reduction` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/triton/delta/count_bucketed_sparse_row_reduction@1/template.yaml`. Original outline: `knowledge/studies/triton/delta/count_bucketed_sparse_row_reduction@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `bucket_cover` (test): Every original row appears once, and every dispatched loop limit covers all of its set bits.
- `bitmap_order` (proof): ffs/clear-lowest enumerates each active contributor once in ascending original order, including empty rows.
- `tile_ownership` (test): Heavy row subtiles and chained feature ranges partition outputs without duplicating row contributions.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
