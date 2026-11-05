# collective_triggered_candidate_queue@1: collective triggered candidate queue

## When to consider

When streaming top-k candidates arrive per lane, buffer improvements locally and merge a whole warp whenever any lane fills its queue. For k=33 choose retained_capacity=64, local_capacity=3. If lane 7 fills its third slot, all 32 lanes merge immediately, including partially filled local queues. Calling add only on valid tail lanes violates the collective premise.

## Computation before and after

When streaming top-k candidates arrive per lane, buffer improvements locally and merge a whole warp whenever any lane fills its queue.

Initialize local and retained queues with losing sentinels. Admit a candidate only if it beats the retained kth threshold. After each lane input, vote whether any local queue has reached local_capacity; if so sort all local queues, merge them with the sorted retained queue and keep retained_capacity winners, clear every local queue, and refresh the kth threshold. Flush pending candidates at end. Block selection additionally synchronizes all warps and merges their sorted queues.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| local_capacity | compile / software | [2, 3, 4, 8] | Compiled per-lane staging capacities observed in the float selector menu. |
| retained_capacity | compile / software | [32, 64, 128, 256, 512, 1024, 2048] | Power-of-two retained warp queue capacity; requested k must fit. |

## Coupled constraints and scope

NVIDIA CUDA warp size 32; source float WarpSelect menu. k=1 is a separate direct reduction. Default Comparator ignores candidate IDs on equal keys; deterministic tie selection requires a compatible comparator and padding discipline. Resource effects: Per-lane local_capacity key/value registers plus retained_capacity pairs per warp; warp path keeps retained queue in registers, block path in shared memory. More capacity trades less frequent merging for more live storage.

Scope predicate: `p.k > 0 and p.k <= retained_capacity and p.full_warp_participation and p.padding_never_wins and p.finite_scores and p.tie_policy_compatible`.

- Precondition `pre_0` (semantic): `p.k > 0 and p.k <= retained_capacity` — Retained queue contains at least requested k slots.
- Precondition `pre_1` (semantic): `p.full_warp_participation and p.padding_never_wins` — Every lane participates in votes/merges, invalid entries never win.
- Precondition `pre_2` (semantic): `p.finite_scores and p.tie_policy_compatible` — Comparators implement requested direction and tie contract; finite input excludes NaN ambiguity.

## Local example and misuse

For a local instance of `map`, identify the object `collective_triggered_candidate_queue` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/faiss/delta/collective_triggered_candidate_queue@1/template.yaml`. Original outline: `knowledge/studies/faiss/delta/collective_triggered_candidate_queue@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Per-lane local_capacity key/value registers plus retained_capacity pairs per warp; warp path keeps retained queue in registers, block path in shared memory. More capacity trades less frequent merging for more live storage.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `no_overflow` (proof): Vote and merge happen before another append to any full lane queue.
- `discard_safe` (proof): Rejected candidates cannot displace the k retained winners under the comparator.
- `collective_complete` (proof): Every admitted candidate participates in the next merge and final flush; full warp participation is maintained even for input tails.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
