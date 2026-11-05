# credit_bounded_deferred_drain@1: credit bounded deferred drain

## When to consider

When producer copies and remote-result drain share one ordered execution stream, delay drain for overlap only while the delay stays below the producer credit-ring depth. contrib/pace/csrc/collective/ag/ag_zero_sm.cu:842-935 sets lag=min(send_times,nvl_ring)-1, not rdma_ring-1.

## Computation before and after

When producer copies and remote-result drain share one ordered execution stream, delay drain for overlap only while the delay stays below the producer credit-ring depth.

For slot n emit input/send copies and local fanout after its producer-slot credit wait. If n>=lag, then emit ready wait, remote-result copies and returned credit for slot n-lag. After final producer slot drain every remaining lagged slot in order. Require lag<ring_depth for matched symmetric schedules so drain0 is enqueued before producer ring_depth stalls awaiting the credit that drain0 returns remotely. Capacity of a different receive ring is not a substitute for this producer-ring bound.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| ring_depth | compile / software | {"lo": 1, "hi": 1024} | Producer slot reuse distance. |
| drain_lag | compile / software | {"lo": 0, "hi": 1023} | Number of later producer slots emitted before draining an earlier remote result. |

## Coupled constraints and scope

Matched copy-stream/network coordinator pipeline with finite ready and credit progress; this breaks a specific queue-order cycle, not a general deadlock theorem. Resource effects: Allows preceding copies to overlap remote latency; lag is coupled to credit capacity.

Scope predicate: `p.same_ordered_stream and p.return_credit_requires_remote_drain and p.rank_schedule_matched and p.drain_lag < p.ring_depth and p.tail_drain_complete and p.ready_wait_before_each_drain`.

- Precondition `pre_0` (semantic): `p.same_ordered_stream and p.return_credit_requires_remote_drain and p.rank_schedule_matched` — A producer stall can block later commands in its own stream; peer credit comes only after their drain.
- Precondition `pre_1` (semantic): `p.drain_lag < p.ring_depth and p.tail_drain_complete and p.ready_wait_before_each_drain` — Delayed consumer remains before first reuse wait that depends on it; readiness and complete tail retained.

## Local example and misuse

For a local instance of `map`, identify the object `credit_bounded_deferred_drain` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/nccl/delta/credit_bounded_deferred_drain@1/template.yaml`. Original outline: `knowledge/studies/nccl/delta/credit_bounded_deferred_drain@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Allows preceding copies to overlap remote latency; lag is coupled to credit capacity.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `liveness` (proof): The earliest mandatory producer reuse wait has the required remote drain queued before its blocking position on every matched stream.
- `cover` (proof): Each producer slot and each drain executes once, including final lag tail.
- `ready` (proof): Drain reads only after network ready and returns credit only after all corresponding copies complete.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
