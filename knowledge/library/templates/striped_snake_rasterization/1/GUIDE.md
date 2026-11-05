# striped_snake_rasterization@1: striped snake rasterization

## When to consider

When persistent/CLC tile traversal should remain near the previous strip endpoint, alternate the major-axis direction between narrow strips. Choose strip width and minor axis; preserve actual final width. For five minor tiles,width2 the final one-tile strip uses width1, not2.

## Computation before and after

A rectangular tile grid is traversed in minor-axis strips while alternating major-axis direction.

Let minor extent be tile_rows for minor_axis=m, otherwise tile_cols; major is the other extent. For strip b with start=b*strip_width, actual width w=min(strip_width,minor-start), its IDs start after all prior full strips. Decode local ID q as minor=start+q%w, major=q//w for even b and major_extent-1-q//w for odd b. Swap axes back if needed. The final short strip uses its actual width in both division and modulo.

Recover the original computation: Erase strip order and alternating direction; Euclidean strip/local decoding yields every original tile exactly once.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| minor_axis | compile / software | ["m", "n"] | Axis grouped into strips. |
| strip_width | compile / software | {"lo": 1, "hi": 65536} | Minor-axis tile width, with a short final strip. |

## Coupled constraints and scope

Independent rectangular tiles, positive strip width and actual-width decoding at the final strip. This adds alternating direction beyond block_rasterization@1 grouped_m.

Scope predicate: `(p.independent_tiles and p.tile_rows >= 1 and p.tile_cols >= 1) and (p.linear_work_ids_cover_exact_grid)`.

- Precondition `pre0` (semantic): `p.independent_tiles and p.tile_rows >= 1 and p.tile_cols >= 1` — Traversal permutes independent output tiles.
- Precondition `pre1` (semantic): `p.linear_work_ids_cover_exact_grid` — Work scheduler supplies every tile ID exactly once.

## Local example and misuse

For a local instance of `work_assign`, identify the object `striped_snake_rasterization` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/triton/delta/striped_snake_rasterization@1/template.yaml`. Original outline: `knowledge/studies/triton/delta/striped_snake_rasterization@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `bijection` (proof): Full and short strips partition the minor domain and alternating major traversal is a permutation.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
