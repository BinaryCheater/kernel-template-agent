# tile_argmin_pair_combine@1: tile argmin pair combine

## When to consider

When distance tiles feed nearest-neighbor output, reduce each tile to value/identity pairs and merge those pairs atomically as one logical state instead of materializing every distance. Tile winners(score2,id7) and(score1,id9) must yield(1,9). Separate unsynchronized stores can produce(1,7), a pair no tile produced. Empty tail tiles offer a losing pair, not ID0 with a real score.

## Computation before and after

When distance tiles feed nearest-neighbor output, reduce each tile to value/identity pairs and merge those pairs atomically as one logical state instead of materializing every distance.

Produce a distance tile and combine norm/metric epilogue while values remain on chip. Reduce each output row over tile columns into one (score,global_candidate_id) using the declared comparator. Merge the local pair with the existing row pair under a row/bin mutex: acquire, read old complete pair, apply comparator, store the selected complete pair, publish writes, release. When one CTA exclusively owns the full candidate row, omit only the global mutex while preserving the same pair reduction. Final output is one winning pair per query row.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| tile_columns | compile / software | {"lo": 1, "hi": 65536} | Candidate columns per producer tile; resource/menu constraints supplied by realization. |

## Coupled constraints and scope

NVIDIA fused distance-NN SIMT and CUTLASS epilogues. Comparator tie and metric convention are workload-bound. A scalar atomicMin on score alone cannot publish a consistent separate ID. Resource effects: Avoids materialized query*candidate distance matrix, retaining tile accumulators and O(query_rows) winner pairs; mutex/bin contention and epilogue reductions replace matrix write/read traffic.

Scope predicate: `p.pair_comparator_total_and_shared and p.invalid_pairs_lose and p.output_initialized_once and p.competing_pair_updates_serialized and p.completion_before_read and p.producers_can_progress`.

- Precondition `pre_0` (semantic): `p.pair_comparator_total_and_shared and p.invalid_pairs_lose` — Tile and global reductions use the same value/identity comparator, with padded pairs losing.
- Precondition `pre_1` (semantic): `p.output_initialized_once and p.competing_pair_updates_serialized` — Each row output starts at a defined losing pair; concurrent value/ID changes are serialized as one logical pair.
- Precondition `pre_2` (semantic): `p.completion_before_read and p.producers_can_progress` — Readers/reset follow all producer updates and locking cannot deadlock residency.

## Local example and misuse

For a local instance of `map`, identify the object `tile_argmin_pair_combine` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cuvs/delta/tile_argmin_pair_combine@1/template.yaml`. Original outline: `knowledge/studies/cuvs/delta/tile_argmin_pair_combine@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Avoids materialized query*candidate distance matrix, retaining tile accumulators and O(query_rows) winner pairs; mutex/bin contention and epilogue reductions replace matrix write/read traffic.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `tile_cover` (proof): Every valid candidate is represented by exactly one producer tile before local reduction.
- `pair_identity` (proof): Winning score and ID always come from the same candidate, including ties.
- `atomic_combine` (proof): Competing CTAs cannot interleave scalar stores into a mixed score/ID pair.
- `fusion_boundary` (proof): All norm/metric transforms required for ranking occur before local argmin, not after pruning changes winners.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
