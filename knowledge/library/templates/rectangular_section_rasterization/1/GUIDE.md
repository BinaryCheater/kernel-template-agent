# rectangular_section_rasterization@1: rectangular section rasterization

## When to consider

When locality requires grouping both tile axes, choose separate M/N section sizes instead of one grouped-axis width. A 2x4 section groups eight nearby output tiles; a short final section must skip padded coordinates. The optimizer varies section sizes while tile extents come from upstream output tiling.

## Computation before and after

A two-dimensional independent tile grid traversed by rectangular sections and then intra-section coordinates.

Create a padded grid of ceil_div(tile_rows,section_m)*ceil_div(tile_cols,section_n) sections. Decode section index in row-major order, then intra-section coordinates in row-major order. Map m=section_row*section_m+local_row,n=section_col*section_n+local_col. Skip coordinates m>=tile_rows or n>=tile_cols. Fuser split/reorder domains realize this traversal; output masks remain the underlying tiled operator obligation.

Recover the original computation: Forget traversal order and padding; every valid output tile has one section and one intra-section coordinate.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| section_m | compile / software | {"lo": 1, "hi": 65536} | Rows per traversal section. |
| section_n | compile / software | {"lo": 1, "hi": 65536} | Columns per traversal section. |

## Coupled constraints and scope

Independent rectangular output tiles, positive section sizes, guarded padded sections. This adds independent M and N section factors beyond block_rasterization@1 grouped_m; it makes no cache-traffic savings guarantee.

Scope predicate: `p.tile_rows >= 1 and p.tile_cols >= 1 and p.independent_tiles and p.padded_coordinates_guarded`.

- Precondition `pre0` (semantic): `p.independent_tiles and p.padded_coordinates_guarded` — Tile permutation is legal and padded section coordinates are not executed as valid outputs.
- Constraint `c0` (semantic): `p.tile_rows >= 1 and p.tile_cols >= 1` — Independent output grid is nonempty.

## Local example and misuse

For a local instance of `work_assign`, identify the object `rectangular_section_rasterization` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/fuser/delta/rectangular_section_rasterization@1/template.yaml`. Original outline: `knowledge/studies/fuser/delta/rectangular_section_rasterization@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Source-bound physical buffer; exact byte count and capacity supplied by binding.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `bijection` (proof): Euclidean division maps every valid tile to exactly one section/local coordinate.

Address/storage transformation preserves values; arithmetic changes require separate obligations.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
