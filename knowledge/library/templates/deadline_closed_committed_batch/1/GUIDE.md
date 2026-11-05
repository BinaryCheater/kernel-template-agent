# deadline_closed_committed_batch@1: deadline closed committed batch

## When to consider

When host requests are coalesced on the GPU under a deadline, distinguish reserved queries from published pointers and close only unreserved slots when time expires. Slot0 reserves just before timeout but publishes after timeout: gather must still wait and copy it. Unreserved slot1 may exit. Treating every null pointer as empty loses slot0; allowing new commits after closure corrupts generation ownership.

## Computation before and after

When host requests are coalesced on the GPU under a deadline, distinguish reserved queries from published pointers and close only unreserved slots when time expires.

Host request first reserves a unique batch row, then publishes its input/output pointers. GPU gather blocks poll their row pointer. Once a row is committed, its block must wait for publication even if deadline passes. If no reservation exists when timeout closes the batch, that row exits without copying and no future reservation may enter this batch generation. Published rows copy into contiguous batch query storage. Enqueue upstream search after gather, scatter results to original output pointers and publish a completion event before request buffers or batch IO slots are reused.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| max_batch | compile / software | {"lo": 1, "hi": 16777215} | Maximum reservation count fitting the source24-bit commit counter; operational launch/storage bounds are tighter. |
| timeout_us | compile / software | {"lo": 1, "hi": 2147483647} | Scheduling timeout, not cancellation permission for committed requests. |

## Coupled constraints and scope

NVIDIA dynamic batching with system-scope pointer atomics and packed commit/closure state. Deadline adjusts batch occupancy, not request cancellation. Source mixed-width token access and host/device ordering require target-specific realization proof. Resource effects: Contiguous batch query/result storage plus pinned request-pointer/token metadata; amortizes search calls at cost of deadline waiting and gather/scatter launches.

Scope predicate: `p.committed_requests_eventually_publish and p.release_acquire_pointer_publication and p.batch_generation_valid and p.request_storage_live and p.uniform_cta_exit and p.gather_precedes_search`.

- Precondition `pre_0` (semantic): `p.committed_requests_eventually_publish and p.release_acquire_pointer_publication` — Every successful reservation eventually submits a valid pointer with matching memory ordering.
- Precondition `pre_1` (semantic): `p.batch_generation_valid and p.request_storage_live` — Ring-buffer token identifies this batch generation and input/output memory stays live through its completion event.
- Precondition `pre_2` (semantic): `p.uniform_cta_exit and p.gather_precedes_search` — Query-copy blocks exit collectively; search is ordered after gather in the same stream.

## Local example and misuse

For a local instance of `map`, identify the object `deadline_closed_committed_batch` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cuvs/delta/deadline_closed_committed_batch@1/template.yaml`. Original outline: `knowledge/studies/cuvs/delta/deadline_closed_committed_batch@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Contiguous batch query/result storage plus pinned request-pointer/token metadata; amortizes search calls at cost of deadline waiting and gather/scatter launches.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `commit_vs_publish` (proof): A missing pointer does not imply an unused slot after reservation, and timeout cannot omit a committed request.
- `close_atomicity` (proof): No new reservation can race past the closed-batch token into a generation being searched/recycled.
- `request_identity` (proof): Each copied row and result scatter refer to the same original request and slot generation.
- `completion` (proof): Request return/reuse follows result scatter, not pointer consumption or search enqueue.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
