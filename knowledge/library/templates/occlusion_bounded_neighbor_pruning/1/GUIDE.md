# occlusion_bounded_neighbor_pruning@1: occlusion bounded neighbor pruning

## When to consider

When graph degree must be bounded while retaining diverse routes, greedily accept ranked neighbors and suppress later candidates whose query distance is dominated by proximity to an accepted neighbor. A query distance10 and accepted-neighbor distance2 yield occlusion5; at alpha1 the candidate is suppressed, at alpha5 it may become eligible if degree remains. Taking reciprocal0.2 makes the opposite decision.

## Computation before and after

When graph degree must be bounded while retaining diverse routes, greedily accept ranked neighbors and suppress later candidates whose query distance is dominated by proximity to an accepted neighbor.

Merge search-visited candidates with existing neighbors, preserving the chosen sorted/deduplicated candidate contract. Track an occlusion value per candidate. Walk candidates in rank order; accept a nonself candidate when its occlusion does not exceed current alpha and degree cap is not reached. Mark accepted entries and update later candidate occlusion to max(old,query_distance/candidate_distance) using the accepted neighbor. Repeat the declared increasing alpha schedule if needed, compact accepted entries and pad unused output slots with invalid IDs.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| max_degree | compile / software | {"lo": 1, "hi": 65536} | Output graph degree cap; algorithm quality parameter, not a free exact-kNN optimization. |

## Coupled constraints and scope

NVIDIA Vamana RobustPrune. Source alpha starts1 and multiplies1.2 until requested bound; that grid is policy, not a theorem of graph quality. Zero inter-candidate distances and duplicate-coordinate IDs need explicit ratio semantics. Resource effects: Candidate-pair distance work grows with accepted neighbors times remaining candidates; cached accepted coordinates amortize reads across a pass. Reuses search distances only when numerical mapping is bit-compatible.

Scope predicate: `p.candidates_ranked_and_ids_valid and p.distances_compatible and p.alpha_schedule_defined and p.approximate_graph_contract and p.collective_acceptance_uniform and p.storage_capacity_sufficient`.

- Precondition `pre_0` (semantic): `p.candidates_ranked_and_ids_valid and p.distances_compatible` — Candidate/query and candidate/candidate distances share the declared metric and numerical convention.
- Precondition `pre_1` (semantic): `p.alpha_schedule_defined and p.approximate_graph_contract` — Caller chooses alpha schedule and admits heuristic route diversity rather than exact closest-degree neighbors.
- Precondition `pre_2` (semantic): `p.collective_acceptance_uniform and p.storage_capacity_sufficient` — CTA agrees on acceptance and synchronizes candidate-coordinate loads and occlusion updates.

## Local example and misuse

For a local instance of `map`, identify the object `occlusion_bounded_neighbor_pruning` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cuvs/delta/occlusion_bounded_neighbor_pruning@1/template.yaml`. Original outline: `knowledge/studies/cuvs/delta/occlusion_bounded_neighbor_pruning@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Candidate-pair distance work grows with accepted neighbors times remaining candidates; cached accepted coordinates amortize reads across a pass. Reuses search distances only when numerical mapping is bit-compatible.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `ranking` (proof): Candidate merge/order/duplicate handling matches the source contract before pruning.
- `occlusion` (proof): Each ratio uses the correct query-distance numerator and accepted-to-candidate denominator, with defined zero/nonfinite policy.
- `cap_and_identity` (proof): At most max_degree nonself valid IDs are emitted and compaction retains exactly accepted entries.
- `barriers` (proof): All candidate-coordinate readers and occlusion writers finish before reuse or next acceptance.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
