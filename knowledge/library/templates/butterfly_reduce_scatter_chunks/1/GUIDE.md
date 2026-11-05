# butterfly_reduce_scatter_chunks@1: butterfly reduce scatter chunks

## When to consider

When each lane holds several outer-reduction statistics, reduce and scatter vector chunks before writing shared partials. Choose subgroup width and vector length; BDIMX=8,NumVals=8 gives four reduction lanes and two final components per lane, reducing shared payload fourfold. Reject component counts not divisible by subgroup width or shuffles crossing another output column.

## Computation before and after

A warp subgroup reduces vector-valued partials while scattering disjoint result chunks among its lanes.

Start each of lanes logical lanes with values partial states. At each descending XOR peer bit halve the live component vector. Lower-half owners keep the lower half and send the upper half; upper-half owners keep the upper half and send the lower half. Combine received peer states componentwise with the retained states. Repeat until each lane owns values/lanes complete states, the contiguous component range [lane*values/lanes,(lane+1)*values/lanes). Publish only these states for subsequent cross-warp combining.

Recover the original computation: Gather the disjoint per-lane output chunks to reconstruct the vector reduction of every original lane partial.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| lanes | compile / software | {"lo": 1, "hi": 65536} | Power-of-two subgroup lanes spanning the reduction axis. |
| values | compile / software | {"lo": 1, "hi": 65536} | Vector components contributed by each subgroup lane. |

## Coupled constraints and scope

Power-of-two warp subgroups with divisible vector length; combine may be a coupled state such as count/mean/M2 provided its merge contract is satisfied. Source BDIMX divides 32, lanes=32/BDIMX and NumVals divisible by lanes.

Scope predicate: `values % lanes == 0 and values >= lanes and p.lanes_power_of_two and p.subgroup_mapping_exact and p.lanes_participate and p.combine_associative_commutative_under_contract and p.valid_partial_states`.

- Precondition `pre0` (semantic): `p.combine_associative_commutative_under_contract and p.valid_partial_states` — The combine and state contract admit the butterfly order.
- Constraint `c0` (semantic): `values % lanes == 0 and values >= lanes and p.lanes_power_of_two` — Each recursive half has an integral chunk.
- Constraint `c1` (semantic): `p.subgroup_mapping_exact and p.lanes_participate` — Peer shuffles stay within the logical subgroup and every required lane participates.

## Local example and misuse

For a local instance of `reduce`, identify the object `butterfly_reduce_scatter_chunks` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/fuser/delta/butterfly_reduce_scatter_chunks@1/template.yaml`. Original outline: `knowledge/studies/fuser/delta/butterfly_reduce_scatter_chunks@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Source-bound physical buffer; exact byte count and capacity supplied by binding.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `component_membership` (proof): Each final component contains each subgroup input exactly once with no component mixing.
- `shuffle_and_state` (test): Physical XOR peers, active mask and any coupled statistical state implement the logical exchange.

Address/storage transformation preserves values; arithmetic changes require separate obligations.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
