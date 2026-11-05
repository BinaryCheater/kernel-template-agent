# iterative_state_residency@2: resident local recurrence state

## When to consider

When a small solve or dependent walk repeatedly materializes its mutable state between steps, keep that state within its owner CTA. Choose registers or shared memory subject to peak live storage; preserve every required per-step output tape and the same transitions, termination and random identity.

## Computation before and after

Retain bounded recurrence state within one CTA through the same transitions, preserving any required output tape.

Load the independent instance initial mutable state once. Execute the same prescribed recurrence, fixed iteration budget or identical termination test, keeping state and current-step temporaries at state_site. A dependent walk may load the next immutable input row selected by its resident state; not all immutable rows need reside simultaneously. Synchronize each cross-thread producer/consumer transition before its reads or overwrite, and reuse workspace after its last reader. In final_only mode store the final state or original explicit failure result once, as in version1. In per_step mode additionally store every required transition output exactly once, with original indices, numerical stage and random identity; external readers wait for its generation. Preserve casts, epsilons, pivot/tie/endpoint policies and update order. No change of solver, path, stopping criterion or proposal law is implied.

Recover the original computation: At each logical iteration boundary, materialize the resident state; this yields exactly the coarse sequence of states under the same recurrence and numerical policy.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| state_site | compile / software | ["registers", "smem"] | Placement of the state; selected realization must support all access sharing and required barriers. |
| output_retention | compile / boundary | ["final_only", "per_step"] | Whether the workload requires only the final state/failure or also an output tape from each transition; this is fixed by consumers, not a tuning freedom. |

## Coupled constraints and scope

Version1 independent bounded matrix normalization/solve scope retained with output_retention=final_only. Version2 adds bounded dependent finite-state walks under the same local state/capacity/termination premises, optionally preserving per-step outputs. DFlash selector uses one warp/request, K<=16, register previous-index/current-K-score state, immutable next-row loads and a Q/token tape. EAGLE greedy verifier uses one thread/request and child/sibling links with an accepted-index tape. External intermediate readers before generation completion remain excluded.

Scope predicate: `p.iteration_dependencies_local_to_cta and p.independent_instances and p.iteration_count_and_numeric_policy_fixed and (output_retention == 'final_only' or (p.required_step_outputs_written_exactly_once and p.step_output_generation_visible_before_external_read))`.

- Precondition `pre_local` (semantic): `p.iteration_dependencies_local_to_cta and p.independent_instances and p.iteration_count_and_numeric_policy_fixed` — One CTA owns every dependency of an instance, including full-row/global-within-instance reductions; iteration count and arithmetic/termination policy are workload inputs.
- Precondition `pre_liveness` (implementation): `p.no_external_intermediate_reader and p.all_local_sync_collectively_reachable` — No consumer requires intermediate states before final publication; every intra-CTA collective is reached by all declared participants.
- Precondition `pre_outputs` (implementation): `output_retention == 'final_only' or (p.required_step_outputs_written_exactly_once and p.step_output_generation_visible_before_external_read)` — Retain every required transition output at the specified stage and identity; consumers wait for the completed generation.
- Constraint `c_capacity` (resource): `p.state_bytes + p.workspace_bytes <= p.selected_site_capacity_bytes` — All simultaneously live iterative state and operation workspace must fit.

## Local example and misuse

For a local instance of `map`, identify the object `resident_local_recurrence_state` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/iterative_state_residency@2/template.yaml`. Original outline: `knowledge/studies/sglang/delta/iterative_state_residency@2/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Mutable state plus peak current-step workspace; immutable inputs and per-step output tape may remain global.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `step` (test): Each local iteration consumes precisely the preceding complete state and implements the same numerical recurrence.
- `lifetime` (test): Shared temporaries are not reused before all required reads and cross-thread values are synchronized.
- `completion` (test): Termination/failure branches and final publication match the coarse contract; every collective remains reachable.
- `step_outputs` (test): Every required transition output and its stage, index, random identity and visibility are preserved.

Fusing kernels may remove casts or change operation order; preserving the stated numerical recurrence is mandatory, otherwise a separate approximation binding is needed. Dependent-walk random draws, NaN/tie selection and CDF endpoints retain their original per-step identity.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
