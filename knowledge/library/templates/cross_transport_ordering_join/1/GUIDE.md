# cross_transport_ordering_join@1: cross transport ordering join

## When to consider

When one logical communication stream uses independent transport queues or engines, establish ordering by joining every active completion domain before a dependent operation crosses domains. contrib/niin/include/niin/sync.h:20-100 drains GIN contexts, TMA, peer-store state; GpuNetIO atomic sidecar is separate domain.

## Computation before and after

When one logical communication stream uses independent transport queues or engines, establish ordering by joining every active completion domain before a dependent operation crosses domains.

Choose ordering or completion contract per operation. One ordered reliable queue to the same peer can preserve put order without draining; when operations rotate across multiple contexts or a separate atomic sidecar, drain every relevant context before crossing to dependent action. Independently drain TMA bulk operations and publish pending peer stores with system fence. Quiet completes all active domains even where fence only needs ordering. Clear pending indicators only after their work is covered. Do not interpret a GIN flush as completing unrelated TMA or atomic-sidecar operations.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Source NIIN fence/quiet semantics and engine tracking. Exact backend domain contracts and collective thread scope must be supplied. Resource effects: May avoid all-context drain for single ordered queue; multi-context load balancing trades ordering overhead.

Scope predicate: `p.operation_domain_membership_recorded and p.completion_primitive_covers_prior_operations and p.same_queue_order_contract_or_all_domains_joined and p.consumer_visibility_fence_matches_scope and p.pending_state_cleared_only_after_successful_completion and p.no_untracked_async_source_reads`.

- Precondition `pre_0` (semantic): `p.operation_domain_membership_recorded and p.completion_primitive_covers_prior_operations` — Track every queue/engine that may carry prior communication, including work issued by cooperating threads when required.
- Precondition `pre_1` (semantic): `p.same_queue_order_contract_or_all_domains_joined and p.consumer_visibility_fence_matches_scope` — Same ordered queue may need only a visibility fence; independent queues require completion joins.
- Precondition `pre_2` (semantic): `p.pending_state_cleared_only_after_successful_completion and p.no_untracked_async_source_reads` — Pending flags cannot discard unfinished operations.

## Local example and misuse

For a local instance of `map`, identify the object `cross_transport_ordering_join` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/nccl/delta/cross_transport_ordering_join@1/template.yaml`. Original outline: `knowledge/studies/nccl/delta/cross_transport_ordering_join@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: May avoid all-context drain for single ordered queue; multi-context load balancing trades ordering overhead.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `domain_cover` (proof): Every prior operation belongs to at least one joined domain; no outstanding source read is omitted.
- `happens_before` (proof): Required local/remote completion plus scoped publication precedes dependent access.
- `minimal_ordering` (proof): A single-queue shortcut is legal only under the same-peer in-order provider contract; multiple unordered queues invalidate it.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
