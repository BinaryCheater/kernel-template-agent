# consumer_complete_scratch_chunking@1: consumer complete reusable chunk scratch

## When to consider

When a multi-kernel computation retains a large intermediate that only nearby consumers need, produce a row chunk, run every consumer, then reuse its scratch. Keep original row identities in masks and metadata, and initialize shared gradient accumulators once across all chunks. This bounds transient storage without requiring one fused kernel; omitting a consumer or rebasing a causal mask as though the chunk were the whole sequence changes the result.

## Computation before and after

Materialize one row chunk and drain all its consumers before reusing intermediate storage.

Replace a full rows-by-intermediate_width temporary with storage for min(chunk_rows,rows) rows. For starts a=0,chunk_rows,... let b=min(a+chunk_rows,rows). Produce the original intermediate for global rows[a,b) into scratch rows[0,b-a). Bind every consumer with both the chunk-local views and the original global row/sequence identity. Run all consumers of that intermediate before the next producer reuses scratch. Write independent outputs to their original[a,b) slices; accumulate shared indexed/reduced outputs using the full invocation's defined initial state. Final partial chunks expose only b-a rows. Preserve masks by explicit original coordinates or a proven equivalent view/offset transformation. Kernel/stream dependencies may establish completion without a host synchronization, but issuing a consumer is not proof it has completed.

Recover the original computation: Concatenate the produced logical chunk intermediates in global row order and compose their consumers, recovering the original full-intermediate dataflow and complete output reductions.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| chunk_rows | runtime / software | {"lo": 1, "hi": 2147483647} | Maximum logical rows materialized in the reusable intermediate; last chunk may be smaller. |

## Coupled constraints and scope

A finite row-independent producer with an explicit complete consumer set and reusable global scratch. Source sparse MLA chunks query tokens, with varlen or batch1 contiguous views, retaining global K/V accumulators. Persistent checkpoints needed by later external passes, cross-chunk producer recurrences and implicit mask-coordinate changes are outside this version. Smaller scratch trades for repeated launches/setup; no speedup is claimed.

Scope predicate: `p.rows >= 1 and p.bytes_per_row >= 1 and p.producer_rows_independent and p.all_consumer_edges_enumerated and p.row_partition_exact_once and p.global_row_identity_preserved_in_every_semantic_consumer and p.chunk_views_and_tail_bounds_valid and p.disjoint_output_slices_or_defined_cross_chunk_reduction and p.shared_accumulators_initialized_once and p.cross_chunk_numeric_order_admitted and p.producer_completion_precedes_each_consumer and p.all_chunk_consumers_complete_before_scratch_reuse and p.no_external_consumer_needs_old_intermediate`.

- Precondition `pre_partition` (semantic): `p.producer_rows_independent and p.all_consumer_edges_enumerated and p.row_partition_exact_once` — Every row belongs to one chunk and all consumers of its intermediate are known; no producer recurrence crosses chunks.
- Precondition `pre_coordinates` (semantic): `p.global_row_identity_preserved_in_every_semantic_consumer and p.chunk_views_and_tail_bounds_valid` — Local chunk coordinates decode original row/head/sequence identity, including causal masks, support selection, metadata and tail stores.
- Precondition `pre_outputs` (semantic): `p.disjoint_output_slices_or_defined_cross_chunk_reduction and p.shared_accumulators_initialized_once and p.cross_chunk_numeric_order_admitted` — Outputs are disjoint slices or explicit complete reductions; shared accumulators are initialized once for the full invocation and changed reduction order is admitted.
- Precondition `pre_lifetime` (implementation): `p.producer_completion_precedes_each_consumer and p.all_chunk_consumers_complete_before_scratch_reuse and p.no_external_consumer_needs_old_intermediate` — Every chunk's producer completes before its consumers, all consumers complete before reuse, and no later external user needs an overwritten intermediate.
- Constraint `c_rows` (semantic): `p.rows >= 1 and p.bytes_per_row >= 1` — This version handles nonempty producer-row domains with positive per-row intermediate storage.
- Constraint `c_capacity` (resource): `min(chunk_rows, p.rows) * p.bytes_per_row <= p.scratch_capacity_bytes` — The largest live chunk fits the scratch allocation.

## Local example and misuse

For a local instance of `store`, identify the object `consumer_complete_reusable_chunk_scratch` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flash-attention/delta/consumer_complete_scratch_chunking@1/template.yaml`. Original outline: `knowledge/studies/flash-attention/delta/consumer_complete_scratch_chunking@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: One intermediate allocation serves every chunk while original output identities remain intact.. Storage: At most min(chunk_rows,rows) times intermediate bytes per row instead of full rows times bytes per row.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: scratch
  level: dram
  scope: kernel
  bytes: min(chunk_rows, p.rows) * p.bytes_per_row
overlap: []
```

## Obligations and numerical checks

- `partition_and_coordinates` (test): Every original row is produced/consumed once, and all masks, indices, sequence boundaries and output coordinates retain their original semantics.
- `consumer_lifetime` (test): Every dependent consumer finishes reading a chunk before its scratch is overwritten, including asynchronously issued consumers.
- `output_composition` (test): Output slices are complete/disjoint or reduced exactly with one initial state and an admitted cross-chunk numerical order.

Chunking can alter cross-row atomic/reduction interleaving; remapping sequence lengths or causal offsets incorrectly changes the mathematical support.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
