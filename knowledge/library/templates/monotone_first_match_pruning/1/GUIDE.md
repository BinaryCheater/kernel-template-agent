# monotone_first_match_pruning@1: monotone first match pruning

## When to consider

When looking for the first matching index, use the best published index to stop workers whose future tiles start later. A relaxed stale bound costs extra work and preserves correctness because best only decreases. Preserve monotone worker order; an arbitrary later-index match cannot justify skipping an earlier unseen tile.

## Computation before and after

An atomic best index bounds increasing tile searches for the first matching input.

Initialize global best=N. Workers visit disjoint increasing index tiles. Before a tile with lower bound L read global best; if best<L, exit because no future index can improve it. Otherwise evaluate the tile predicate, reduce to its earliest match, atomically minimize global best and exit that worker on a found tile. A stale relaxed best may postpone pruning but cannot suppress an improving match. Sentinel N means no match after all workers complete.

Recover the original computation: Erase skipped ranges whose indices exceed a witnessed matching index; remaining atomic minima equal the earliest matching original input.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

First-index queries over immutable inputs with deterministic predicate, disjoint increasing worker tiles and a globally atomic minimum. Does not justify pruning arbitrary find-any or max-value searches.

Scope predicate: `(p.request_is_first_matching_index and p.tile_indices_increase_per_worker and p.predicate_deterministic) and (p.best_initialized_to_length and p.best_updates_atomic_min and p.tile_search_finds_local_first)`.

- Precondition `pre0` (semantic): `p.request_is_first_matching_index and p.tile_indices_increase_per_worker and p.predicate_deterministic` — Search requests a minimum index and every worker future range has a known monotone lower bound.
- Precondition `pre1` (semantic): `p.best_initialized_to_length and p.best_updates_atomic_min and p.tile_search_finds_local_first` — Published best is always a valid found index or sentinel.

## Local example and misuse

For a local instance of `map`, identify the object `monotone_first_match_pruning` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cccl/delta/monotone_first_match_pruning@1/template.yaml`. Original outline: `knowledge/studies/cccl/delta/monotone_first_match_pruning@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `safe_pruning` (proof): Every skipped index is strictly greater than an already found valid matching index.
- `coverage` (test): Every potentially improving index is searched unless a smaller valid match was already published.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
