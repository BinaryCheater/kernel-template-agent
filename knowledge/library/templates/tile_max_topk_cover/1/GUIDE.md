# tile_max_topk_cover@1: topk over tiled row

## When to consider

When exact top-k scans a large row but k is small, retain the k highest-maximum tiles and select from their complete contents. The attained maxima certify rank coverage without a heuristic threshold. Choose tile width and retained count subject to the witness bound; ties may choose any valid identities.

## Computation before and after

Attained tile maxima supply rank witnesses for whole-tile top-k pruning.

Compute an attained exact maximum for each disjoint tile. Keep retained_tiles tiles with largest maxima. Gather their complete valid contents, retaining original indices, and run exact top-k over this union. An omitted value cannot exceed its tile maximum; when tiles are omitted, at least k selected distinct tiles each supply an actual value no smaller than that maximum. Hence the selected union contains some allowed global top-k. Sorting retained tile IDs for locality does not change that union. A separate full-row sum or normalization still visits every input; selected tiles alone do not preserve logsumexp.

Recover the original computation: Reinsert each omitted tile: every omitted value is dominated or tied by at least k distinct attained maxima already in retained tiles.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| tile_width | compile / software | [1024] | Elements per source tile; final tile is masked. |
| retained_tiles | runtime / software | {"lo": 1, "hi": 2147483647} | Number of highest-maximum tiles whose complete valid contents enter final selection. |

## Coupled constraints and scope

Source LiLiCorr uses width1024, retained_tiles=min(k,ceil(V/1024)); its Triton branch requires this count to be a power of two. Tile-ID sorting and repeated argmax implement gather and final selection. Source finite -3e38 padding requires valid selected winners strictly above that sentinel; global smallest-index tie identity is not established.

Scope predicate: `retained_tiles >= min(p.k, p.tile_count) and retained_tiles <= p.tile_count and p.k > 0 and p.k <= p.length and p.tiles_partition_valid_row and p.nan_inputs_excluded and p.tile_maxima_exact_and_attained and p.selected_tiles_are_highest_maxima and p.complete_selected_tiles_gathered and p.final_selector_exact_topk and p.arbitrary_tie_identity_admitted and p.padding_never_selected`.

- Precondition `pre_row` (semantic): `p.k > 0 and p.k <= p.length and p.tiles_partition_valid_row and p.nan_inputs_excluded` — Nonempty disjoint tiles partition all valid finite ordered values.
- Precondition `pre_max` (semantic): `p.tile_maxima_exact_and_attained and p.selected_tiles_are_highest_maxima` — Each tile summary is its attained maximum, not an arbitrary upper bound.
- Precondition `pre_select` (semantic): `p.complete_selected_tiles_gathered and p.final_selector_exact_topk and p.arbitrary_tie_identity_admitted and p.padding_never_selected` — Gather every valid value in selected tiles, preserve original identities and allow arbitrary boundary ties.
- Constraint `c_cover` (semantic): `retained_tiles >= min(p.k, p.tile_count) and retained_tiles <= p.tile_count` — Keep at least one tile-maximum witness for each required rank, or keep all tiles.

## Local example and misuse

For a local instance of `reduce`, identify the object `topk_over_tiled_row` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/tile_max_topk_cover@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/tile_max_topk_cover@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One maximum per tile and retained_tiles*tile_width gathered values.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `rank_cover` (proof): The retained union contains an allowed global top-k under arbitrary tied identity.
- `gather_identity` (test): All valid members of each selected tile are gathered once at original coordinates; padding cannot win.

NaNs or a finite padding sentinel that can win invalidate rank comparisons. Full-row logsumexp remains a separate reduction over all tiles.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
