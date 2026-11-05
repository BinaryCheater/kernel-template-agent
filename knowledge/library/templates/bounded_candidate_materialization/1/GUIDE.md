# bounded_candidate_materialization@1: bounded candidate materialization

## When to consider

When boundary candidates may exceed scratch capacity, keep original identities and fall back to replaying the current prefix predicate over original input. Choose capacity without discarding overflow candidates. Radix top-k may scan the original array again when previous_len exceeds buffer_length; it must not process only the truncated buffer.

## Computation before and after

A filter pass uses compact candidate storage when it fits, otherwise replays its predicate over immutable original inputs.

At a refinement pass maintain an active membership predicate over original item identity. If the previous active count fits capacity and its compact buffer is complete, scan buffered keys plus original indices. Otherwise scan original input and reevaluate the same active predicate. Apply the next filter/histogram only to active items. Preserve original indices when fetching values. The first pass uses original inputs; later passes may switch representation without changing membership.

Recover the original computation: Erase the selected physical input representation; every pass sees exactly the active original occurrences.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| capacity | compile / software | {"lo": 1, "hi": 65536} | Maximum candidate identities held by the temporary buffer. |

## Coupled constraints and scope

Iterative selection/refinement whose active set is defined by a compact replayable predicate and immutable original input. Candidate buffer capacity is a resource decision, not a semantic bound on input length.

Scope predicate: `(p.original_inputs_immutable and p.active_predicate_replayable and p.original_identity_preserved) and (p.buffer_exact_when_selected and p.capacity_test_uses_actual_candidate_count)`.

- Precondition `pre0` (semantic): `p.original_inputs_immutable and p.active_predicate_replayable and p.original_identity_preserved` — Original values and identities can be filtered again with exactly the current prefix predicate.
- Precondition `pre1` (semantic): `p.buffer_exact_when_selected and p.capacity_test_uses_actual_candidate_count` — Buffer fallback never truncates or mistakes capacity for candidate count.

## Local example and misuse

For a local instance of `work_assign`, identify the object `bounded_candidate_materialization` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cccl/delta/bounded_candidate_materialization@1/template.yaml`. Original outline: `knowledge/studies/cccl/delta/bounded_candidate_materialization@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `membership_identity` (proof): Buffered and replayed paths enumerate the same original active occurrences exactly once.
- `capacity_and_lifetime` (test): Only complete in-capacity buffers are read and original data remain live through every replay.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
