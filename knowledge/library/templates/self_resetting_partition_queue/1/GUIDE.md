# self_resetting_partition_queue@1: partition local dynamic ticket queue with retirement reset

## When to consider

For independent task partitions with repeated ordered launches, let workers claim complete tasks dynamically and reset the ticket counter when the last claimer retires. Keep retirement separate from output completion and give concurrent launches separate counter epochs.

## Computation before and after

Partition-local atomic task tickets with two-lookahead claims and last-claimer counter reset.

For partition t, start next[t]=0 and retired[t]=0 with exactly P_t workers. Each worker leader claims two initial tickets with atomic fetch-add, stages valid task inputs and shares its tickets within the CTA. For each valid current task, claim one further lookahead ticket, compute and publish that complete task, and advance the private lookahead queue; out-of-range tickets do not access task payload. Stop when the next task is invalid, after draining every valid ticket already owned. Each leader then performs one device-scope acquire-release atomic increment of retired[t], wrapping at P_t. The leader observing previous count P_t-1 resets next[t]=0; the wrapping increment already resets retired[t]=0. No claimer may access next[t] after retirement. The retirement protocol protects counter reuse only; full output visibility and the next invocation still wait for whole-kernel completion. Reuse the now-zero buffer on the same completion-ordered stream/graph generation, or use separate storage for concurrent launches. Dynamic task assignment never changes the arithmetic inside a task.

Recover the original computation: Order successful in-range ticket claims by their atomic sequence. They give a bijection to tasks0..N-1, with each task produced once from unchanged inputs. Erase out-of-range claims and retirement/reset operations after their final use; completed kernel leaves both banks at the original zero state for the next ordered invocation.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

FlashInfer deterministic zero-history width4 causal-convolution forward, source f5497330bfd286211643, SM100/103/107. Partition is channel tile; token chunk has n_sub*8 rows and actual grid gives tile-specific CTA count. Two int32 words per tile start/end zero; atomic retired counter is unsigned wrapping increment, ticket word is signed fetch-add. Reuse removes a separate per-call counter clear under a whole-kernel stream dependency. It is not an inter-CTA output publication barrier, an overlapping epoch protocol or a generic persistent service queue.

Scope predicate: `(p.partition_tasks_independent_complete_and_uniquely_named) and (p.two_counter_banks_zero_initialized_and_integer_claim_bound_safe) and (p.actual_partition_claimer_count_matches_atomic_wrap_limit) and (p.claimer_has_issued_all_counter_accesses_before_retirement) and (p.atomic_retirement_chain_orders_last_claimer_reset_after_all_claimers) and (p.next_invocation_orders_after_whole_kernel_completion_and_workspace_is_exclusive) and (p.partition_geometry_and_counter_bank_offsets_coherent) and (p.native_atomic_scope_and_prefetch_pipeline_supported)`.

- Precondition `pre_items` (semantic): `p.partition_tasks_independent_complete_and_uniquely_named` — Each partition has independent uniquely indexed tasks; one successful ticket owns a complete output task. Task arithmetic and inputs are independent of which worker claims it, so dynamic completion order does not define a floating reduction order.
- Precondition `pre_counter` (implementation): `p.two_counter_banks_zero_initialized_and_integer_claim_bound_safe` — Each active partition has an atomic next-ticket word and a retirement word, both zero before its first launch. With P claimers and N valid tasks, source two-lookahead policy performs N+2P total fetch-add claims, which must fit its nonnegative signed ticket range and address arithmetic.
- Precondition `pre_members` (implementation): `p.actual_partition_claimer_count_matches_atomic_wrap_limit` — Actual grid has P_t=ceil((B-t)/T) claimers for partition t. Exactly one leader per claimer takes tickets and retires once; zero-work claimers still retire. Retirement wraps at P_t, not at a uniform capacity count or number of live tasks.
- Precondition `pre_retire` (implementation): `p.claimer_has_issued_all_counter_accesses_before_retirement` — A worker retires only after issuing its last fetch-add claim and will never use the ticket counter again in this invocation. In-flight prefetched valid tasks remain its responsibility before local completion. Retirement is not a promise that every output store by every CTA thread has completed.
- Precondition `pre_reset` (implementation): `p.atomic_retirement_chain_orders_last_claimer_reset_after_all_claimers` — Device-scope acquire-release wrapping increment serializes retirement; the leader observing the last previous count resets next-ticket to zero after all claimers have left that counter. Only that leader resets it, and the wrapping retirement word itself returns to zero.
- Precondition `pre_epoch` (implementation): `p.next_invocation_orders_after_whole_kernel_completion_and_workspace_is_exclusive` — Counters and tensor inputs stay owned through the whole kernel. Reuse is ordered after complete kernel execution on the same stream or an equivalent event dependency, not merely after the retirement count wraps. Concurrent streams/graphs use separate counter storage or explicit serialization.
- Precondition `pre_geometry` (implementation): `p.partition_geometry_and_counter_bank_offsets_coherent` — Ticket bank offset t and retirement bank offset T+t match the same active partition count and actual grid. A later geometry may reuse the buffer only after prior completion, with sufficient all-zero active capacity. Captured launches retain their own valid counter epoch.
- Precondition `pre_backend` (implementation): `p.native_atomic_scope_and_prefetch_pipeline_supported` — Concrete source is SM100/103/107 deterministic Cake width4 depthwise-convolution forward, using one CTA leader and two claimed lookahead tasks per CTA. Native memory scopes, double-buffer input transfers, task tails and release ordering need the selected backend implementation.

## Local example and misuse

For a local instance of `work_assign`, identify the object `partition_local_dynamic_ticket_queue_with_retirement_reset` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/self_resetting_partition_queue@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/self_resetting_partition_queue@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Zero-initialized next-ticket/retirement banks per partition, plus CTA-local lookahead and staged task buffers.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `task_coverage` (test): All in-range tickets are consumed exactly once despite lookahead and arbitrary worker interleaving; invalid tickets have no payload access.
- `reset_order` (proof): The reset occurs only after every possible counter claimer retires, with exact participant count and no later claim from the current epoch.
- `reuse_epoch` (test): Whole-kernel completion separates invocations; a zero retirement word is not mistaken for completed outputs or permission for concurrent reuse.
- `native_pipeline` (test): Ticket broadcast, input staging, CTA synchronization and tail computation preserve the declared owner-local task recipe.

Dynamic assignment preserves output bits only when each task owns its complete arithmetic. Using retirement wrap as output completion or resetting while a prefetched claimer is active mixes epochs; a wrong participant count can reset tickets early.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
