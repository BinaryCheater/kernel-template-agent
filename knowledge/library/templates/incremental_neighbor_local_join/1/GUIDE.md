# incremental_neighbor_local_join@1: incremental neighbor local join

## When to consider

When a neighbor graph changes incrementally, compare new-new and new-old neighborhoods while avoiding repeated old-old joins, then merge candidate updates under destination ownership. New{2,5},old{7} examines pair(2,5),(2,7),(5,7) and excludes pivot/self. Dropping(5,7) misses a new-old opportunity; reusing old-old is valid only under the algorithm provenance premise.

## Computation before and after

When a neighbor graph changes incrementally, compare new-new and new-old neighborhoods while avoiding repeated old-old joins, then merge candidate updates under destination ownership.

For each pivot gather forward/reverse sampled new and old neighbors into separate shared lists and remove duplicates. Tile source features or codes once, compute new-new pair distances and new-old pair distances, exclude self edges, then offer improving neighbors to destination graph lists. Mark/age newly processed entries for future iterations. Reuse existing old-old work rather than rebuilding all pairs. A destination segment lock serializes loads, sorted insertion/deduplication and stores of competing candidate updates; release only after publication.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| sample_capacity | compile / software | [32, 64] | Bounded sampled forward/reverse neighborhood per source binding; limits search coverage, not merely storage. |

## Coupled constraints and scope

NVIDIA NN-descent graph construction with bounded samples. SIMT retains input dtype then float arithmetic; FP16 WMMA converts operands and accumulates float; quantized bitwise/dp4a/u4 paths require their own decoder/factor mapping. These are score-policy variants, not automatically numerically equivalent. Resource effects: Gathered sample tile shared across O(sample_capacity^2) comparisons, avoiding old-old repeated pair work; segment locks and sampled graph state add storage/synchronization. No convergence/recall guarantee from local join alone.

Scope predicate: `p.new_old_partition_defined and p.sample_ids_unique and p.old_old_revisit_not_required and p.approximate_graph_contract and p.distance_policy_defined and p.graph_merge_serialized`.

- Precondition `pre_0` (semantic): `p.new_old_partition_defined and p.sample_ids_unique` — New and old sampled candidates have defined provenance and duplicates are removed before joining.
- Precondition `pre_1` (semantic): `p.old_old_revisit_not_required and p.approximate_graph_contract` — Graph-construction algorithm permits omitting previously processed old-old candidate pairs.
- Precondition `pre_2` (semantic): `p.distance_policy_defined and p.graph_merge_serialized` — SIMT/tensor or quantized score contract is explicit; competing destination neighbor updates use required exclusion/publication.

## Local example and misuse

For a local instance of `map`, identify the object `incremental_neighbor_local_join` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cuvs/delta/incremental_neighbor_local_join@1/template.yaml`. Original outline: `knowledge/studies/cuvs/delta/incremental_neighbor_local_join@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Gathered sample tile shared across O(sample_capacity^2) comparisons, avoiding old-old repeated pair work; segment locks and sampled graph state add storage/synchronization. No convergence/recall guarantee from local join alone.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `candidate_pairs` (proof): Visited join pairs match new-new/new-old sampled sets, preserving pair identities and excluding self edges.
- `distance` (proof): Every score matches the declared precision/quantized metric, with padding excluded.
- `merge` (proof): Concurrent updates cannot lose an accepted edge or violate destination list ordering/duplicate policy.
- `coverage_scope` (proof): Sampling/aging omissions are admitted graph-construction approximation, not exact kNN proof.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
