# iterative_state_residency@1: resident iteration state

## When to consider

When every dependency of a small iterative solve belongs to one CTA, retain its full state in registers or shared memory across the prescribed recurrence. This removes repeated global state traffic and launches at the cost of peak resident storage and local synchronization. Keep iteration budget, arithmetic and failure semantics fixed; residency does not certify convergence.

## Computation before and after

Retain iterative state within one CTA across a fixed recurrence rather than materializing each phase globally.

Load an independent instance initial state once. Execute the original fixed recurrence for its declared iteration budget or identical termination test, retaining state and temporaries at state_site. Synchronize every cross-thread producer/consumer transition before the next phase reads or overwrites values. Reuse workspace only after its last reader. Store the final state or the original explicit failure result once. Preserve all original casts, epsilons, pivot policies and update order; no change of solver or convergence criterion is implied.

Recover the original computation: At each logical iteration boundary, materialize the resident state; this yields exactly the coarse sequence of states under the same recurrence and numerical policy.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| state_site | compile / software | ["registers", "smem"] | Placement of the state; selected realization must support all access sharing and required barriers. |

## Coupled constraints and scope

Small independent iterative matrix normalization or solve whose full working set fits oneCTA. Sources: HCxHC Sinkhorn normalization with register state; NCxNV IPM with shared state and block Cholesky. This rule concerns residence and synchronization only; it does not prove convergence or optimality.

Scope predicate: `p.iteration_dependencies_local_to_cta and p.independent_instances and p.iteration_count_and_numeric_policy_fixed`.

- Precondition `pre_local` (semantic): `p.iteration_dependencies_local_to_cta and p.independent_instances and p.iteration_count_and_numeric_policy_fixed` — One CTA owns every dependency of an instance, including full-row/global-within-instance reductions; iteration count and arithmetic/termination policy are workload inputs.
- Precondition `pre_liveness` (implementation): `p.no_external_intermediate_reader and p.all_local_sync_collectively_reachable` — No consumer requires intermediate states before final publication; every intra-CTA collective is reached by all declared participants.
- Constraint `c_capacity` (resource): `p.state_bytes + p.workspace_bytes <= p.selected_site_capacity_bytes` — All simultaneously live iterative state and operation workspace must fit.

## Local example and misuse

For a local instance of `map`, identify the object `resident_iteration_state` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/iterative_state_residency@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/iterative_state_residency@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Complete mutable state plus peak local workspace.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `step` (test): Each local iteration consumes precisely the preceding complete state and implements the same numerical recurrence.
- `lifetime` (test): Shared temporaries are not reused before all required reads and cross-thread values are synchronized.
- `completion` (test): Termination/failure branches and final publication match the coarse contract; every collective remains reachable.

Fusing kernels may remove casts or change operation order; preserving the stated numerical recurrence is mandatory, otherwise a separate approximation binding is needed.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
