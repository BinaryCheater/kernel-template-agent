# prefix_ready_contraction@1: contraction operand ready frontier

## When to consider

When an operand is produced in increasing reduction-axis order, let its consumer start on a fully published prefix while the tail is still being written. Wait separately before tail reads, and include accumulator correction in the first readiness condition. At K128 the source publishes P columns0..95 first, then96..127; K192 instead publishes0..127 first. A full-buffer-ready event loses this overlap, while treating prefix-ready as whole-buffer-ready permits uninitialized tail reads. Native barrier/fence ordering remains required.

## Computation before and after

Consume a complete contraction prefix before the operand suffix is ready.

Partition the reduction axis into [0,prefix_k) and [prefix_k,K). The producer writes and publishes the complete first region, then continues the suffix. Consumer waits for prefix readiness together with any initial-accumulator dependency, issues only prefix MMA atoms into the accumulator, then waits for the suffix's generation-specific readiness before issuing its atoms into that same accumulator. Publish and acquire operand writes using the backend's required memory/proxy fences and barrier protocol. Retirement of the entire asynchronous read sequence, not the prefix signal, permits operand reuse. If an existing accumulator must be rescaled, that correction finishes before prefix execution and never scales new prefix/suffix contributions afterward.

Recover the original computation: Wait for the complete operand before executing the identical prefix-then-suffix contraction atom sequence. With the same accumulator initialization and ordering, deleting the overlap changes timing only.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| prefix_k | compile / software | {"lo": 1, "hi": 1024} | Contraction reduction extent whose complete operand payload is published first; suffix is derived as K-prefix_k. |

## Coupled constraints and scope

One-CTA asynchronous contractions with an independently writable two-region operand and aligned reduction-atom split. Source TMEM P is published around three quarters of K, rounded down to32 columns. A two-CTA implementation requires a separately verified cluster publication/arrival binding. Prefix readiness alone cannot authorize reading the tail or changing the accumulator.

Scope predicate: `p.prefix_suffix_partition_complete and p.consumer_mapping_matches_frontier and p.prefix_payload_visible_before_publish and p.suffix_payload_visible_before_publish and p.accumulator_ready_before_prefix and p.generation_owned_and_not_lapped and p.payload_reuse_after_all_reads and p.other_operand_stable and p.single_cta_readiness_scope and p.wait_protocol_supported and p.producer_can_finish_suffix and p.same_accumulation_order_admitted`.

- Precondition `pre_cover` (semantic): `p.prefix_suffix_partition_complete and p.consumer_mapping_matches_frontier` — The ready prefix includes every row/feature read by prefix instructions, with no missing fragment or overlap in reduction coverage.
- Precondition `pre_ready` (implementation): `p.prefix_payload_visible_before_publish and p.suffix_payload_visible_before_publish and p.accumulator_ready_before_prefix` — Each signal orders its own payload; the initial accumulator and any correction are ready before the first prefix MMA.
- Precondition `pre_lifetime` (semantic): `p.generation_owned_and_not_lapped and p.payload_reuse_after_all_reads and p.other_operand_stable` — Signals and payload belong to the requested generation; neither operand is overwritten while asynchronous MMA still reads it.
- Precondition `pre_progress` (implementation): `p.single_cta_readiness_scope and p.wait_protocol_supported and p.producer_can_finish_suffix and p.same_accumulation_order_admitted` — The suffix producer can progress while the consumer waits; backend fences/completion and the split MMA order are admitted.
- Constraint `c_split` (implementation): `p.instruction_k > 0 and prefix_k < p.K and prefix_k % p.instruction_k == 0 and p.K % p.instruction_k == 0` — Nonempty prefix/suffix split at an MMA reduction-atom boundary.

## Local example and misuse

For a local instance of `sync`, identify the object `contraction_operand_ready_frontier` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flash-attention/delta/prefix_ready_contraction@1/template.yaml`. Original outline: `knowledge/studies/flash-attention/delta/prefix_ready_contraction@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: Prefix computation overlaps suffix production.. Storage: One operand generation with two publication regions and a shared accumulator.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `frontier_mapping` (proof): Every prefix/suffix instruction reads only the corresponding published region, and their reduction atoms cover K exactly once.
- `readiness_and_correction` (test): Operand publication and accumulator readiness jointly precede their dependent instructions, with correct generation and native memory ordering.
- `retirement_and_progress` (test): No payload reuse precedes the last async reader, and waiting for suffix publication does not prevent its producer from running.

Changing prefix/suffix issue order or rescaling after new contributions changes the contraction's numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- alternative_of: per_slot_handshake (outside menu); A whole-slot full event waits for all payload; this frontier permits early prefix consumption and needs a distinct suffix wait and full retirement.

Structured contract: [template.yaml](template.yaml).
