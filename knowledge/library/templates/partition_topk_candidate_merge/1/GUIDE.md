# partition_topk_candidate_merge@1: partition topk candidate merge

## When to consider

When candidates are partitioned across indexes or devices, merge at most k winners per partition with explicit global identity translation and completion before final selection. Two partitions have local IDs0,1 but bases0 and100: winners(0,score1) and(0,score2) represent global0 and100. Last batch of3 queries must use stride3 per rank, not the allocated maximum8. Translating invalid -1 to99 creates a false valid candidate.

## Computation before and after

When candidates are partitioned across indexes or devices, merge at most k winners per partition with explicit global identity translation and completion before final selection.

Compute local top-k pairs for each partition, retaining actual valid count and original score. Translate local IDs with partition base or mapping, never translating invalid sentinels as data. Gather pairs into query-major rows using the current batch size, not a maximum-batch stride. Select global k from their union; a tree may repeatedly merge pairs of partition groups under the same order. For max-close metrics negate scores only under a matching sign/return convention. Index search approximation remains upstream; this merge does not restore candidates already omitted there.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| partitions | compile / software | {"lo": 1, "hi": 1048576} | Candidate partitions/shards; can be a scheduling choice only if union contract stays fixed. |

## Coupled constraints and scope

NVIDIA multi-index/composite, multi-GPU sharded and tiered candidate merging. Exactness is relative to the declared candidate set and sound local selection; ANN local search supplies an approximate set, not exact global nearest neighbors. Duplicate records across replicated partitions need deduplication or an explicitly multiplicity-preserving contract. Resource effects: O(partitions*k) candidates per query plus transfer/temporary buffers; root vs tree merge trades aggregate root work for intermediate selections and communications.

Scope predicate: `p.partitions_cover_declared_candidates and p.local_winners_sound and p.identity_translation_injective and p.invalid_sentinels_preserved and p.metric_comparators_compatible and p.all_partitions_complete`.

- Precondition `pre_0` (semantic): `p.partitions_cover_declared_candidates and p.local_winners_sound` — Every declared candidate belongs to the intended partition and local pruning retains all candidates needed under the agreed top-k order.
- Precondition `pre_1` (semantic): `p.identity_translation_injective and p.invalid_sentinels_preserved` — Local IDs translate to unique global identities, while invalid slots stay invalid and cannot win.
- Precondition `pre_2` (semantic): `p.metric_comparators_compatible and p.all_partitions_complete` — Every local/merge stage uses compatible metric and tie policy; inputs finish writing before merge.

## Local example and misuse

For a local instance of `map`, identify the object `partition_topk_candidate_merge` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cuvs/delta/partition_topk_candidate_merge@1/template.yaml`. Original outline: `knowledge/studies/cuvs/delta/partition_topk_candidate_merge@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: O(partitions*k) candidates per query plus transfer/temporary buffers; root vs tree merge trades aggregate root work for intermediate selections and communications.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `local_pruning` (proof): Every discarded partition candidate is dominated by k retained candidates in that partition under the final comparator.
- `identity` (proof): Translation is one-to-one over valid IDs and invalid padding never enters rank output.
- `layout` (proof): Query/partition/rank offsets agree between producers and consumer for full and partial batches.
- `completion` (proof): Cross-stream/device reads follow all producer writes, including intermediate tree stages.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
