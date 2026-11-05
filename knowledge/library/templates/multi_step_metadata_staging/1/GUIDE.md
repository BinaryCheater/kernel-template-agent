# multi_step_metadata_staging@1: grouped future step metadata window

## When to consider

When per-step scales are contiguous along K but scalar staging wastes transactions, stage several future scales together and publish one window. Keep each scale tied to its own partial and retain the window until the last slice/feature owner retires. Metadata batching is independent of raw-dot grouping.

## Computation before and after

Amortize metadata transfers/publications over several arithmetic steps without changing their individual semantics.

Let G=steps_per_window and S=stages. For logical output tile u and step group g, reserve the next metadata window slot according to the shared global window ordinal. Copy every row of FP32 metadata for steps[g*G,(g+1)*G) into slice-addressable shared memory; row-major staged offset=slot*(rows*G)+row*G+j. Publish the complete window once. For j=0..G-1, wait for that arithmetic step partial independently, read metadata at row*G+j, and apply it to that step only under the original arithmetic/scale contract. Release each arithmetic partial by its own pipeline, but release the metadata window only after every j and every metadata reader has retired. Advance metadata slot/phase once per group. Preserve or consistently reset the shared ordinal across output tiles. G1 recovers per-step metadata publication. Partial groups require a separate tail path and are outside this version.

Recover the original computation: Expand each vector metadata copy into G scalar-step copies and replace each slice reference by the corresponding original metadata element. The arithmetic steps, scale ownership and outputs are unchanged; only transfer grouping and metadata lifetime differ.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| steps_per_window | compile / software | [1, 4] | Source menu: stage one step or four contiguous FP32 scale steps; larger grouping changes transfer/sync only. |
| stages | compile / software | {"lo": 1, "hi": 64} | Number of grouped metadata ring slots, subject to explicit backend/resources; source setup uses10 atG1 or3 atG4. |

## Coupled constraints and scope

Pinned contiguous grouped FP8 GEMM on SM100-family GPUs,128-deep arithmetic partials, K-major FP32 SFA and per-N-group SFB. Source batches4 scale steps when K/128 is divisible4, otherwise1. Its3/10-stage source rings have independent arithmetic accumulator pipelines. Metadata format/rows and vector transfers are bound; other batch widths, partial grouped tails or changed scale grouping need another binding.

Scope predicate: `p.steps >= 1 and p.steps % steps_per_window == 0 and p.metadata_rows >= 1 and (p.metadata_step_axis_contiguous_and_vector_aligned and p.window_sources_and_destinations_same_logical_rows_steps_expert_generation) and (p.each_math_step_reads_its_own_window_slice_and_arithmetic_partition_unchanged) and (p.whole_window_visible_before_any_slice_consumer and p.copy_completion_accounts_for_full_window_bytes) and (p.window_not_released_until_all_step_slices_and_readers_retire) and (p.producer_consumer_window_ordinal_slot_phase_agree_across_output_tiles and p.resource_teardown_and_final_drain_valid) and (p.selected_batch_vector_width_stage_count_and_resources_supported and p.source_metadata_immutable_through_copy)`.

- Precondition `pre_layout` (implementation): `p.metadata_step_axis_contiguous_and_vector_aligned and p.window_sources_and_destinations_same_logical_rows_steps_expert_generation` — Each row has consecutive metadata for the grouped steps, with true vector alignment and exact row/expert identity.
- Precondition `pre_consumption` (semantic): `p.each_math_step_reads_its_own_window_slice_and_arithmetic_partition_unchanged` — Grouping transfers never combines differently scaled raw arithmetic partials or substitutes one step scale for another.
- Precondition `pre_publication` (implementation): `p.whole_window_visible_before_any_slice_consumer and p.copy_completion_accounts_for_full_window_bytes` — Every grouped copy completes before the first slice is read; full completion covers all metadata bytes.
- Precondition `pre_retire` (implementation): `p.window_not_released_until_all_step_slices_and_readers_retire` — The empty signal occurs only after the last of G step consumers and all feature owners are done, not after the first scalar load.
- Precondition `pre_ring` (implementation): `p.producer_consumer_window_ordinal_slot_phase_agree_across_output_tiles and p.resource_teardown_and_final_drain_valid` — Ring state advances once per window, with matching generation across tile transitions and final drain.
- Precondition `pre_backend` (implementation): `p.selected_batch_vector_width_stage_count_and_resources_supported and p.source_metadata_immutable_through_copy` — The target supports4*G-byte copies and the allocated ring; immutable inputs survive asynchronous reads.
- Constraint `c_complete` (implementation): `p.steps >= 1 and p.steps % steps_per_window == 0 and p.metadata_rows >= 1` — This source projection covers complete groups only; an incomplete K tail selects G1 rather than dropping metadata.

## Local example and misuse

For a local instance of `transfer`, identify the object `grouped_future_step_metadata_window` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/multi_step_metadata_staging@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/multi_step_metadata_staging@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: FP32 row-by-step metadata windows with one full/empty generation per batch.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: metadata_windows
  level: smem
  scope: cta
  bytes: 4 * stages * steps_per_window * p.metadata_rows
overlap: []
```

## Obligations and numerical checks

- `slice_identity` (test): Grouped row/step offsets match the original scale metadata for each expert and output tile.
- `separate_arithmetic` (test): Every arithmetic partial retains its own scale; batching metadata never delays promotion across different scale groups.
- `window_lifetime` (test): Publication precedes every slice read and release follows all step/owner readers, including ring wrap and final tile.
- `vector_alignment_tail` (test): Copy alignment/width is legal, grouped metadata is complete and unsupported tails select a valid alternative.

Moving a scale application outside several unequal-scale partials changes the operator. This rule changes metadata placement only; native arithmetic order remains the original contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
