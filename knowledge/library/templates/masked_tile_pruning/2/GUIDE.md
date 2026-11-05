# masked_tile_pruning@2: tile mask class

## When to consider

A structural causal/window mask leaves whole key tiles with zero contribution, and the remaining tiles can be classified without changing any valid score.


## Computation before and after

Exact interval restriction with two independently masked boundaries.

Obtain normalized metadata from the workload mask and selected tile shape: total_tiles, start, leading, core, trailing and end=start+leading+core+trailing. Mask kind/alignment/window/sequence lengths are workload metadata, not an optimizer's freedom to change the operator. Omit [0,start) and [end,total_tiles) only after certifying every omitted element is identity for every visible row. Classify [start,start+leading) and [end-trailing,end) as masked edges and the intervening core as unmasked. For mask_all_kept apply the exact element mask throughout [start,end); for peel_boundaries apply it only on the two edges. Ascending traversal visits left edge/core/right edge; descending visits right edge/core/left edge, each with reversed tile order. Zero-length segments issue no iterations. For an empty kept interval produce the explicitly admitted identity/consumer policy. Negative raw helper counts are rejected; this rule does not silently clamp or reinterpret them.

Recover the original computation: Reinsert omitted all-identity tiles and exact element masks in the certified core; under the admitted traversal policy the reduction has the same visible result.

## Bind the rule

1. Caller fixes sequence length S, causal relation j<=i and any window. Choose query/key tile dimensions elsewhere. Derive interval metadata from the actual visible query rows, not the padded tile extent.
2. For a query tile with rows [q0,q1) intersected with [0,S), causal equal-length attention keeps key tiles [0,ceil(q1/BN)). Thus total_tiles=ceil(S/BN), start=0 and end=ceil(q1/BN). Every j>=q1 is masked for all visible rows.
3. A tile is core only if all its key coordinates satisfy j<=q0 and j<S. Compute leading/core/trailing counts from actual intervals, and check c_counts. All other kept tiles require the exact element predicate j<S and j<=i; invalid query rows are never stored.
4. Start with masking=mask_all_kept and traversal=ascending. Choose peel_boundaries only after proving every visible core element valid. Both producers and consumers must use the same end and visit no empty segments.


| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| masking | compile / software | ["mask_all_kept", "peel_boundaries"] | Apply the exact element mask to every kept tile or omit mask instructions only for the certified unmasked core. |
| traversal | compile / software | ["ascending", "descending"] | Visit the same classified interval in either admitted reduction order; source FMHA selects ascending. |

## Coupled constraints and scope

Exact structural masking whose kept tile interval has at most two masked edges and a certified unmasked core. Counts and mask kind are open workload metadata; their classification requires its own evidence. Source FMHA uses ascending leading/core/trailing loops; mask_all_kept and descending are algebraic choices, not compiled variants claimed here. Raw negative counts and unproved empty output policies do not bind this version. Value-dependent nonidentity pruning remains a different rule with an explicit approximation certificate.

Scope predicate: `p.tile_partition_exact and p.counts_integral_and_representable and p.mask_metadata_stable and p.mask_value_is_identity and p.outside_interval_all_identity and p.element_mask_exact and p.padded_output_rows_unobserved and (masking == 'mask_all_kept' or p.core_all_visible_elements_valid) and p.traversal_numerically_admitted and p.empty_interval_policy_admitted and p.loop_and_producer_ranges_match`.

- Precondition `pre_partition` (implementation): `p.tile_partition_exact and p.counts_integral_and_representable and p.mask_metadata_stable` — Tile indices are a complete exact partition and the geometry-derived boundaries fit the implementation.
- Precondition `pre_skip` (semantic): `p.mask_value_is_identity and p.outside_interval_all_identity` — Every skipped element for every visible output row is a reduction identity.
- Precondition `pre_mask` (semantic): `p.element_mask_exact and p.padded_output_rows_unobserved` — Masked-loop code applies the complete predicate including residual bounds; padded output rows are never observed.
- Precondition `pre_core` (semantic): `masking == 'mask_all_kept' or p.core_all_visible_elements_valid` — Omitting mask code in the core requires every element of every visible row to be valid.
- Precondition `pre_order` (implementation): `p.traversal_numerically_admitted and p.empty_interval_policy_admitted and p.loop_and_producer_ranges_match` — Reduction order and empty intervals are admitted and all producer/consumer roles visit matching work.
- Constraint `c_counts` (semantic): `p.total_tiles >= 0 and p.start >= 0 and p.leading >= 0 and p.core >= 0 and p.trailing >= 0 and p.end == p.start + p.leading + p.core + p.trailing and p.end <= p.total_tiles` — Nonnegative normalized segment counts partition one bounded kept interval.
- Constraint `c_empty` (semantic): `p.end > p.start or p.empty_interval_result_defined` — Omitting every tile requires an explicit empty-reduction result and compatible downstream handling.

## Local example and misuse

S=513, BM=32, BN=64, query tile rows [64,96): total_tiles=9, start=0,end=2,leading=0,core=1,trailing=1. Key tile0 (j=0..63) is valid for every row; tile1 needs j<=i; tiles2..8 are skipped. The last query tile has one visible row i=512: end=9,core=8,trailing=1, and its final key tile admits only j=512. A classification using padded q1=544 without clipping is incorrect.


## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/masked_tile_pruning@2/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/masked_tile_pruning@2/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Skipping fully masked tiles reduces both score and value work; peeling avoids per-element mask instructions in the core. Classification and role synchronization add complexity. Mask-all-kept is the simple legal starting point; it still skips certified outside tiles and preserves tail checks.


```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `segment_partition` (exhaustive): The three normalized segments are disjoint and cover exactly the kept interval in either traversal.
- `skipped_fully_masked` (exhaustive): Every omitted tile is identity-only for every visible row.
- `unmasked_really_unmasked` (exhaustive): Every core element of every visible row is valid whenever mask instructions are omitted.
- `mask_applied` (test): The chosen element-mask backend sets exactly invalid elements to the identity including physical tail coordinates.
- `role_and_empty_consistency` (test): Producers and consumers agree on kept ranges and an empty interval has the declared initialized result and downstream policy.

Skipping identities is exact; changing traversal or handling zero-mass downstream normalization still needs its own numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- requires: [reduction_partition@1](../../reduction_partition/1/GUIDE.md), [reduction_partition@2](../../reduction_partition/2/GUIDE.md); The tile partition defines the exact units classified by the mask.

Structured contract: [template.yaml](template.yaml).
