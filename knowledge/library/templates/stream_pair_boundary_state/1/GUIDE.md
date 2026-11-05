# stream_pair_boundary_state@1: position tagged pair carry across packed invocations

## When to consider

When pairwise compression receives independently advancing requests in arbitrary chunks, retain the unmatched even token with its absolute position. Compute outputs against old carry before committing the new tail. Match both stream identity and position; chunk length parity alone does not identify a pair.

## Computation before and after

Carry one unmatched even token across arbitrary packed stream chunk boundaries.

For each nonempty request chunk with starting absolute position a, positions are a+i. Emit only odd positions. An odd token after the first chunk row reads its immediately preceding even row from the same request. An odd first row reads old carry only if its tag equals a-1; otherwise emit no pair. Each valid pair invokes the declared processor on its even/odd payloads and retains the odd row output/destination. After all old-carry readers finish, examine the last position: if even, copy that last payload/gate to the stream state and tag it with that absolute position; if odd, invalidate the tag. Empty or inactive requests preserve state. With parallel_read_then_commit use a grid/launch-wide read-complete boundary before commit; an ordinary CTA-local barrier cannot order different output/commit CTAs. A subsequent invocation waits for the complete payload/tag commit. State-ID reuse first invalidates the old stream generation.

Recover the original computation: Replace every accepted carry by its tagged original even token. Every emitted odd row then has exactly the adjacent absolute even/odd pair that an unsplit stream processor would consume; forgetting chunk boundaries recovers that pair stream. Missing/discontinuous predecessors intentionally do not emit.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| execution | compile / software | ["serial_per_request", "parallel_read_then_commit"] | Process a request serially with one carry owner, or parallelize output rows against immutable old carry and commit each request tail afterward. The native source uses two ordered launches; serial is an inferred alternative. |

## Coupled constraints and scope

Pinned B12x SM120/121 CSA2 compression uses FP32 value/gate payloads with512 channels, even/odd absolute pairs, BF16 pooled checkpoint, weighted FP32 RMSNorm and BF16 output. The rule chooses state/phase organization; pair math is an explicit input. Native two-launch commit is source-backed; a serial per-request backend is inferred. General ratio>2, overlapping windows, duplicate live state IDs and concurrent mutation are excluded.

Scope predicate: `p.max_tokens >= 1 and p.max_requests >= 1 and p.max_states >= 1 and p.payload_features >= 1 and (p.live_request_ranges_valid_disjoint_and_in_capacity and p.positions_contiguous_within_each_request) and (p.live_state_ids_unique_per_invocation_and_stable_across_continuations and p.recycled_state_slots_invalidated_before_new_stream) and (p.pending_payload_has_absolute_even_position_tag and p.tag_payload_generation_matches) and (p.pair_processor_and_required_rounding_checkpoints_explicit and p.pair_inputs_finite_and_numerics_admitted) and (p.old_carry_readers_complete_before_pending_state_overwrite and p.commit_payload_visible_before_next_invocation) and (p.only_valid_odd_positions_emit_and_output_slot_metadata_matches and p.nonemitting_output_rows_defined) and (p.selected_execution_implemented_and_resources_fit and p.mutable_state_and_outputs_exclusive_until_invocation_finishes)`.

- Precondition `pre_segments` (semantic): `p.live_request_ranges_valid_disjoint_and_in_capacity and p.positions_contiguous_within_each_request` — Packed request ranges have exact starts/ends and absolute consecutive positions; empty ranges remain empty.
- Precondition `pre_identity` (semantic): `p.live_state_ids_unique_per_invocation_and_stable_across_continuations and p.recycled_state_slots_invalidated_before_new_stream` — Carry is keyed by one stable logical stream; a recycled ID cannot inherit another stream even if position tags match.
- Precondition `pre_tag` (semantic): `p.pending_payload_has_absolute_even_position_tag and p.tag_payload_generation_matches` — Pending value/gate belongs to its recorded absolute even position; an odd boundary accepts only tag=position-1.
- Precondition `pre_processor` (implementation): `p.pair_processor_and_required_rounding_checkpoints_explicit and p.pair_inputs_finite_and_numerics_admitted` — The same declared pair operator consumes adjacent values/gates regardless of whether the first comes from carry or current rows.
- Precondition `pre_read_commit` (implementation): `p.old_carry_readers_complete_before_pending_state_overwrite and p.commit_payload_visible_before_next_invocation` — Old carry is an immutable read snapshot until every current reader finishes; ordered commit publishes the next payload/tag before reuse.
- Precondition `pre_output` (semantic): `p.only_valid_odd_positions_emit_and_output_slot_metadata_matches and p.nonemitting_output_rows_defined` — Emitted outputs remain at the odd token row with its destination metadata; absent/invalid pairs have the declared zero/invalid emission representation.
- Precondition `pre_backend` (implementation): `p.selected_execution_implemented_and_resources_fit and p.mutable_state_and_outputs_exclusive_until_invocation_finishes` — Execution supports the selected serial or parallel protocol and has disjoint live mutable storage.
- Constraint `c_capacity` (implementation): `p.max_tokens >= 1 and p.max_requests >= 1 and p.max_states >= 1 and p.payload_features >= 1` — Prepared capacities cover token rows, request rows, stable state IDs and payload features.

## Local example and misuse

For a local instance of `map`, identify the object `position_tagged_pair_carry_across_packed_invocations` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/stream_pair_boundary_state@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/stream_pair_boundary_state@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Per stable stream ID, one payload/gate pair and absolute position tag.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `chunk_invariance` (test): All chunkings of a contiguous stream emit the same absolute odd-position pairs and preserve an unmatched final even token.
- `tag_identity` (test): Position discontinuities, empty requests and recycled state IDs cannot bind a carry from the wrong stream/position.
- `read_commit_order` (test): No tail write invalidates a pending value while a current first-odd row is reading it; next invocations see complete committed payloads.
- `processor_and_emission` (test): Processor rounding, output row and emitted slot are identical across in-chunk and carry pairs.

Chunk invariance assumes the same pair operator and casts. Moving the pooled BF16 checkpoint past RMSNorm or using scalar rather than per-channel gates changes the source operator.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
