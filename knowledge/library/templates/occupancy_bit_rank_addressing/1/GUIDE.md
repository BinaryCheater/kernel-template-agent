# occupancy_bit_rank_addressing@1: occupancy bit rank addressing

## When to consider

When requested sparse pixel coordinates are unique and sorted within each tile, encode membership in bits and recover the packed record address by population-count rank. SparseTileLayout.cu:132-165 and RasterizeSparseAddressing.cuh:49-74. Duplicate-coordinate behavior is not certified by inspected source checks.

## Computation before and after

When requested sparse pixel coordinates are unique and sorted within each tile, encode membership in bits and recover the packed record address by population-count rank.

For tile position s, set bit s%W of word s//W. A reader returns invalid if bit is clear; otherwise rank is popcount of all earlier words plus popcount(word & ((1<<(s%W))-1)); read pixel_map[tile_start+rank]. Use width-safe masks for zero and boundary positions. Duplicate coordinates must be rejected or deduplicated before constructing this representation.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| word_bits | compile / software | {"lo": 32, "hi": 64} | Legal unsigned occupancy word width; source uses uint32. |

## Coupled constraints and scope

Unique sparse coordinates; this is membership/rank, not multiplicity-preserving compaction. Resource effects: One bit per tile pixel plus packed mapping; rank work over preceding words.

Scope predicate: `p.unique_in_tile_positions and p.sorted_packed_records and p.zero_initialized_words and p.tile_ownership_exclusive and p.mask_and_map_published`.

- Precondition `pre_0` (semantic): `p.unique_in_tile_positions and p.sorted_packed_records and p.zero_initialized_words` — One set bit corresponds to exactly one record in ascending pixel-position order.
- Precondition `pre_1` (semantic): `p.tile_ownership_exclusive and p.mask_and_map_published` — Single tile writer ORs bits; readers see matching map generation.

## Local example and misuse

For a local instance of `map`, identify the object `occupancy_bit_rank_addressing` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/gsplat/delta/occupancy_bit_rank_addressing@1/template.yaml`. Original outline: `knowledge/studies/gsplat/delta/occupancy_bit_rank_addressing@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One bit per tile pixel plus packed mapping; rank work over preceding words.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `bijection` (proof): Ascending set-bit rank is a bijection from requested positions to packed records.
- `ownership` (proof): Masks start zero and have one tile owner; publication precedes all readers.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
