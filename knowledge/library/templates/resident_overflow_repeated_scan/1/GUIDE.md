# resident_overflow_repeated_scan@1: resident overflow repeated scan

## When to consider

When repeated radix passes outgrow resident shared memory, keep a prefix or suffix window and re-stream the rest through bounded slots. Choose resident capacity and stream slot count; derive final visit order from the tie policy. Reversing histogram sweeps can reuse turnaround chunks, while deterministic filtering must still select original indices in its requested order.

## Computation before and after

A repeated predicate scan keeps a fixed input window resident and re-streams the remaining immutable chunks.

Partition each worker input chunks into resident window and overflow. Load resident data once. For each iterative radix/predicate pass, consume resident values and stream all overflow through bounded reusable slots, applying the same current predicate. Resident histogram work may overlap the first overflow load. Source alternates overflow traversal direction between passes so turnaround chunks already in slots need not be reloaded. If final ties require ascending original indices, retain the low-index first-visited window; descending ties retain the high-index window. Final filtering observes that required original-coordinate order rather than slot arrival order.

Recover the original computation: Forget resident/stream slot placement and traverse each original occurrence once under the pass predicate; the same histogram or selected set is obtained.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| resident_chunks | compile / software | {"lo": 1, "hi": 65536} | Input chunks retained across repeated passes. |
| stream_slots | compile / software | {"lo": 1, "hi": 65536} | Reusable temporary chunk slots for overflow scans. |

## Coupled constraints and scope

Repeated scans over immutable chunks with a fixed retained window and bounded asynchronous overflow slots. Histogram passes are order-independent; deterministic final filtering requires explicit original-index ordering. No arbitrary reversal of an ordered scan is allowed.

Scope predicate: `(p.resident_and_overflow_partition_input and p.original_inputs_immutable) and (p.current_predicate_applied_to_both_regions and p.original_coordinates_preserved) and (p.stream_slot_completion_before_read_and_reuse and p.resident_storage_fits)`.

- Precondition `pre0` (semantic): `p.resident_and_overflow_partition_input and p.original_inputs_immutable` — Resident and streamed chunks cover the original input once per pass.
- Precondition `pre1` (semantic): `p.current_predicate_applied_to_both_regions and p.original_coordinates_preserved` — All regions apply the same active predicate and retain original identities.
- Precondition `pre2` (semantic): `p.stream_slot_completion_before_read_and_reuse and p.resident_storage_fits` — Asynchronous loads complete before consumption and every slot retires before refill.

## Local example and misuse

For a local instance of `work_assign`, identify the object `resident_overflow_repeated_scan` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cccl/delta/resident_overflow_repeated_scan@1/template.yaml`. Original outline: `knowledge/studies/cccl/delta/resident_overflow_repeated_scan@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `pass_cover` (proof): Each pass consumes all resident and overflow chunks once with matching predicates and no stale slot generation.
- `tie_order` (test): Deterministic final traversal preserves requested original-index preference independent of streaming order.
- `slot_lifetime` (test): Load completion, consumption and next-generation refill of every stream slot are ordered.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
