# ranked_detour_neighbor_pruning@1: ranked detour neighbor pruning

## When to consider

When a ranked neighbor row must shrink, count two-hop alternatives through earlier neighbors and retain low-detour edges before adding reverse or protected connectivity edges. Ranked candidates B0,B1 where an earlier D also reaches B1 give counts0,1, favoring B0. Equal counts choose earlier rank. A protected edge must survive even if its detour count would otherwise rank poorly.

## Computation before and after

When a ranked neighbor row must shrink, count two-hop alternatives through earlier neighbors and retain low-detour edges before adding reverse or protected connectivity edges.

For each source A cache its ranked neighbors B. For each earlier neighbor D, inspect D outgoing neighbors and increment the count of any matching later A->B edge. Penalize self edges, saturate counts and mark invalid IDs losing. Repeatedly choose minimum (detour_count,original_rank), emit its node and invalidate every duplicate occurrence. Optional graph composition first reserves explicitly protected edges, then fills remaining slots with nonduplicate pruned and reverse edges without evicting protected edges.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| output_degree | compile / software | {"lo": 1, "hi": 65534} | Output route budget; source packs candidate position into 16 bits, so input degree must also fit. |

## Coupled constraints and scope

NVIDIA CAGRA graph optimization. This is a route-selection heuristic, not proof of nearest-neighbor recall or global connectivity unless the protected-edge mechanism establishes that separately. Resource effects: One cached neighbor row plus counters per warp and two-hop adjacency scans; outputs fewer edges. Reverse graph construction requires atomic per-destination reservations and its own capacity/merge premises.

Scope predicate: `p.ranked_graph_rows and p.valid_or_marked_neighbors and p.input_degree_fits_16bit and p.enough_unique_valid_neighbors and p.approximate_route_contract and p.protected_edges_fit`.

- Precondition `pre_0` (semantic): `p.ranked_graph_rows and p.valid_or_marked_neighbors` — Input neighbor order is defined and invalid IDs are distinguished from real graph nodes.
- Precondition `pre_1` (semantic): `p.input_degree_fits_16bit and p.enough_unique_valid_neighbors` — Tagged minimum representation is valid and enough distinct candidates exist for requested degree.
- Precondition `pre_2` (semantic): `p.approximate_route_contract and p.protected_edges_fit` — Detour pruning is an admitted graph quality heuristic; any protected connectivity edges fit the degree cap.

## Local example and misuse

For a local instance of `map`, identify the object `ranked_detour_neighbor_pruning` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cuvs/delta/ranked_detour_neighbor_pruning@1/template.yaml`. Original outline: `knowledge/studies/cuvs/delta/ranked_detour_neighbor_pruning@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One cached neighbor row plus counters per warp and two-hop adjacency scans; outputs fewer edges. Reverse graph construction requires atomic per-destination reservations and its own capacity/merge premises.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `detour_count` (proof): Each admitted A->D->B match increments the intended later direct-edge counter exactly once under input duplicate policy.
- `selection` (proof): Tagged minimum preserves count priority then source rank and rejects invalid/self entries under declared policy.
- `unique_output` (proof): Repeated output contains no duplicate node and has the requested degree or explicit insufficient-candidate failure.
- `protected_capacity` (proof): Connectivity-preserving composition retains every designated protected edge and cannot overflow the degree budget.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
