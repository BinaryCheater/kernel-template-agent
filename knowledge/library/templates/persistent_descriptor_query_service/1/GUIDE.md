# persistent_descriptor_query_service@1: persistent descriptor query service

## When to consider

When repeated small query batches should share resident CTAs, hand each worker a published job descriptor and retain it until all assigned query outputs have completed. A two-query job sets completion only on the second arrival. If worker0 resets its mailbox after reading query0, host may send another assignment but must not overwrite the first descriptor while worker1 still computes query1.

## Computation before and after

When repeated small query batches should share resident CTAs, hand each worker a published job descriptor and retain it until all assigned query outputs have completed.

Each persistent CTA polls its own worker handle. Thread0 observes a job/query assignment, resets the handle to wait-for-work and performs a system acquire fence. A warp loads the descriptor blob into shared storage and the CTA synchronizes. Execute the query against descriptor input/output pointers. Every thread publishes its output writes with system release ordering, then synchronize. Thread0 increments the descriptor completion count; the final query sets the job completion flag. A new worker assignment must not mutate a still-live descriptor. A no-more-work handle exits collectively.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| descriptor_capacity | compile / software | [8192] | Observed maximum descriptor slots; query assignment itself is runtime. |

## Coupled constraints and scope

NVIDIA CAGRA single-CTA persistent query service. Actual mailbox publisher must be checked for matching ordering and buffer retirement before instantiation. Search algorithm and its recall remain separate from service protocol. Resource effects: Resident CTAs avoid repeated kernel launches; descriptor copies, per-worker polls, per-query completion atomics and persistent residency consume resources. No measured crossover inferred.

Scope predicate: `p.descriptors_published_before_handle and p.system_scope_fences and p.exact_query_assignment and p.descriptor_not_reused_early and p.counter_initialized_and_no_wrap and p.stop_sentinel_agreed`.

- Precondition `pre_0` (semantic): `p.descriptors_published_before_handle and p.system_scope_fences` — Host/device publisher establishes descriptor visibility before assigning a worker handle; readers use matching system memory ordering.
- Precondition `pre_1` (semantic): `p.exact_query_assignment and p.descriptor_not_reused_early` — Each query completes once and descriptors/results remain live until all workers retire.
- Precondition `pre_2` (semantic): `p.counter_initialized_and_no_wrap and p.stop_sentinel_agreed` — Per-job completion count starts valid and shutdown sentinel is recognized by every resident worker.

## Local example and misuse

For a local instance of `map`, identify the object `persistent_descriptor_query_service` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cuvs/delta/persistent_descriptor_query_service@1/template.yaml`. Original outline: `knowledge/studies/cuvs/delta/persistent_descriptor_query_service@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Resident CTAs avoid repeated kernel launches; descriptor copies, per-worker polls, per-query completion atomics and persistent residency consume resources. No measured crossover inferred.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `descriptor_lifetime` (proof): Resetting a worker handle frees assignment mailbox only, not the descriptor or result buffers.
- `complete_visibility` (proof): Job completion becomes observable only after all assigned queries publish their results.
- `exact_assignment` (proof): One query arrival corresponds to one finished query, including empty/output-filtered queries.
- `shutdown` (proof): All CTA threads take the same stop path and cannot leave peers at a barrier.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
