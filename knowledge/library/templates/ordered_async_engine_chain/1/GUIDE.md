# ordered_async_engine_chain@1: ordered async engine chain

## When to consider

When a tensor-core engine can consume a prior engine copy in issue order, chain copy→MMA→next copy and wait once at the external observation boundary. This can reuse tensor operand storage without an SM wait between operations. Preserve proxy fences and final commit/wait; a TMA transfer from another engine needs its own readiness join.

## Computation before and after

Operations on one ordered asynchronous engine reuse intermediate storage without an intervening SM-side completion wait.

Issue shared-to-tensor copy, tensor-core MMA consuming that tensor data, and optionally a later tensor copy overwriting the same intermediate. The common engine executes them in issue order, so the later overwrite cannot precede the earlier MMA read. Track completion for the whole sequence with one engine commit to a barrier and wait only before SM-side consumption or incompatible external access. Generic shared writes before the engine copy require proxy fencing; an mbarrier wait after completion may supply the source retirement boundary before generic reuse.

Recover the original computation: Insert a completion wait after each engine operation; ordered engine semantics and final wait make the same values visible at all external observations.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Blackwell tcgen05 copy/MMA/commit sequence with matching engine execution scope and legal shared/tensor layouts. Does not order arbitrary TMA, SM arithmetic or other CTAs merely because the operations are asynchronous.

Scope predicate: `(p.same_ordered_engine_and_scope and p.engine_operations_preserve_issue_order) and (p.shared_inputs_proxy_published and p.external_accesses_ordered_at_chain_boundaries) and (p.intermediate_storage_live_through_chain and p.commit_wait_before_external_result_use)`.

- Precondition `pre0` (semantic): `p.same_ordered_engine_and_scope and p.engine_operations_preserve_issue_order` — All chained operations have the same engine/group scope and required engine execution order.
- Precondition `pre1` (semantic): `p.shared_inputs_proxy_published and p.external_accesses_ordered_at_chain_boundaries` — SM generic writes/reads and other engines synchronize at the chain boundaries.
- Precondition `pre2` (semantic): `p.intermediate_storage_live_through_chain and p.commit_wait_before_external_result_use` — Storage remains allocated and external consumers wait actual committed completion.

## Local example and misuse

For a local instance of `sync`, identify the object `ordered_async_engine_chain` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/triton/delta/ordered_async_engine_chain@1/template.yaml`. Original outline: `knowledge/studies/triton/delta/ordered_async_engine_chain@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `engine_dependency` (proof): Every chained consumer executes after its engine producer and before a later destructive engine write.
- `external_visibility` (test): Non-engine accesses observe only completed/proxy-published values at chain boundaries.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
