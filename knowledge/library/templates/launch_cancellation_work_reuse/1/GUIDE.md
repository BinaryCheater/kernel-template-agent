# launch_cancellation_work_reuse@1: cancelled launch work owner

## When to consider

When the full grid already names cooperative work groups and active groups finish at different times, consider reusing a worker by cancelling an unstarted group. Preserve exclusive launch ownership, protect the response through all readers, query coordinates only after success and stop after an observed failure. The pinned CLC route has one response slot with224 readers for one CTA or416 for two; its unconditional coordinate-query IR still needs guarded-lowering evidence or a corrected implementation before reuse.

## Computation before and after

Reuse an active worker for an exclusively cancelled not-yet-dispatched launch group.

A natively dispatched worker first processes its own launch group's work. Its controller acquires the one response slot and submits a cancellation request for a not-yet-dispatched group. After asynchronous response completion, readers first inspect success. On success only, decode the cancelled launch identity, translate it to the cooperative group's work and retain all local-role offsets; release the response after every reader has finished and process that work. On failure, terminate this worker's reuse loop without decoding coordinates or issuing another request after that failure is observed. Uncancelled pending groups retain their native launch ownership.

Recover the original computation: Replace each successfully cancelled identity's reused iteration by the native launch group it removed; preserve each logical group's work and erase responses and worker reuse.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| participants | compile / boundary | {"lo": 1, "hi": 4096} | Number of response-reader arrivals protecting the single shared cancellation response before reuse. |

## Coupled constraints and scope

Finite launches of cooperative work groups that may reuse an active worker to process a group cancelled before native dispatch. Cancellation ownership differs from a software atomic-counter queue: failed cancellation need not imply that every pending group is absent. Native launch remains responsible for uncancelled work. This version has one response slot, all-reader release and no speculative queries following an observed failure. Physical launch count and latency are unpriced.

Scope predicate: `p.independent_groups and p.cancel_removes_launch and p.success_ids_unique and p.success_checked_before_coordinates and p.stop_after_observed_failure and p.slot_published_before_read and p.slot_reused_after_all_readers and p.group_decode_bijective and p.group_launch_fair`.

- Precondition `pre_response` (implementation): `p.success_checked_before_coordinates and p.stop_after_observed_failure` — Failed responses are not decoded and no new query follows an observed failure.
- Precondition `pre_slot` (semantic): `p.slot_published_before_read and p.slot_reused_after_all_readers` — All participating roles see the same response before the producer can overwrite it.
- Precondition `pre_work` (semantic): `p.independent_groups and p.group_decode_bijective and p.group_launch_fair` — Independent native launch identities decode to the intended work groups and remaining uncancelled groups eventually dispatch.
- Constraint `c_readers` (semantic): `participants == p.response_readers and p.response_slots == 1` — This single-slot version waits for exactly the actual reader arrivals.
- Constraint `c_backend` (implementation): `p.cancel_removes_launch and p.success_ids_unique` — A successful cancellation removes one not-yet-dispatched group and returns an exclusively owned work identity.

## Local example and misuse

For a local instance of `work_assign`, identify the object `cancelled_launch_work_owner` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/launch_cancellation_work_reuse@1/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/launch_cancellation_work_reuse@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: Active group resources survive across multiple logical launch identities.. Storage: One shared response protected until all readers release it.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: response
  level: smem
  scope: cta
  bytes: p.response_bytes
overlap: []
```

## Obligations and numerical checks

- `exclusive_owner` (exhaustive): Every launch identity is either natively dispatched or successfully cancelled once and assigned to exactly one active worker group.
- `response_lifetime` (test): Every required role observes the same completed response before its shared slot is overwritten.
- `failure_guard` (exhaustive): No coordinate query uses a failed response and no cancellation request is issued after that failure is observed.
- `target_progress` (unspecified): Backend cancellation and native dispatch implement the stated ownership and fairness premises for the selected launch.

Reassigns independent cooperative groups; within-group reduction and arithmetic ordering remain separate.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
