# state_independent_chunk_factors@1: parallel feedback factor preparation before serial state chain

## When to consider

When the serial state chain is short on parallel work, prepare its Q/K/gate-only chunk factors independently and store low-rank W/P records. Keep values unmaterialized until the chain, publish the workspace before consumption, and preserve exact chunk/state and native layout identities.

## Computation before and after

Parallel preparation of state-independent low-rank chunk factors before an ordered recurrent consumer.

In on_chain, form feedback factors and corrected residual R at the sequential state consumer under its declared arithmetic. In prepared_workspace, prepare independent chunk/head records EK,EQ,W=K_restore^T*T,P=C*T,E_end without reading H or V, encode each record for the later native readers, finish async stores and publish the complete workspace. Then each sequence/head/value shard walks its chunk prefix in order. With H_operand=the declared conversion of incoming H, form X=EK*H_operand and Z=V-X, output Y=(EQ*H_operand+P*Z)*resolved_query_scale, and H_next=diag(E_end)*H+W*Z under the declared state arithmetic. Capture projections from old H before replacing it. Retain original V instead of storing value-dependent U/O factors. Read-only factors may serve multiple disjoint value shards. Retire every reader before reusing prepared storage. Whole-workspace publication here is an ordered launch boundary, not concurrent chain-first execution.

Recover the original computation: Substitute W=K_restore^T*T and P=C*T. Associativity recovers R=T*(V-EK*H_operand), then original output and state contractions under the admitted reassociation/cast policy. Preparing factors early is legal because they do not depend on H or V.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| factor_supply | compile / software | ["on_chain", "prepared_workspace"] | Compute chunk feedback next to its state consumer, or prepare independent chunks in parallel and publish all factor records before the serial state chain. |

## Coupled constraints and scope

KDA BT16 SM100-family decomp prepares all chunk/head factors, then one chain CTA per sequence/head/value-half (64 of128 value rows). The fused engine is the on-chain alternative. Runtime route n_seq*heads*2<=SM_count is a source policy only. No concurrent flag-ring publication, generic parallel prefix scan, arbitrary chunk widths or guaranteed performance gain is claimed.

Scope predicate: `p.chunk_tokens == 16 and p.key_features == 128 and p.chunk_head_records >= 1 and p.factor_element_bytes == 2 and p.diagonal_element_bytes == 4 and (p.verified_chunk_feedback_factors_and_boundary_defined) and (p.prepared_factors_independent_of_incoming_state_and_values) and (p.factor_inputs_metadata_and_original_values_have_coherent_generations) and (p.workspace_encoder_and_all_native_consumer_maps_match) and (p.prep_completes_before_chain_and_all_workspace_readers_retire_before_reuse) and (p.unique_sequence_state_shards_and_ordered_chunk_consumption) and (p.folded_factor_casts_and_state_operand_recipe_admitted) and (p.factor_supply_width_target_dtype_and_workspace_capacity_supported)`.

- Precondition `pre_factors` (implementation): `p.verified_chunk_feedback_factors_and_boundary_defined` — Each chunk has state-independent EK, EQ, E_end, K_restore, causal C and weighted feedback T with R=T*(V-EK*H). H_next=diag(E_end)*H+K_restore^T*R and Y=EQ*H+C*R under the admitted chunk feedback contract.
- Precondition `pre_independence` (implementation): `p.prepared_factors_independent_of_incoming_state_and_values` — Prepare only W=K_restore^T*T and P=C*T plus EK,EQ,E_end; these depend on matching Q/K/gates/beta and normalization parameters, not incoming H or original V. No precomputed state-dependent residual may be reused for another H.
- Precondition `pre_generations` (implementation): `p.factor_inputs_metadata_and_original_values_have_coherent_generations` — Q/K/gates/beta/normalization and sequence/chunk order remain coherent through preparation and all readers; original V remains available to the chain. Changed inputs require regeneration before reuse.
- Precondition `pre_layout` (implementation): `p.workspace_encoder_and_all_native_consumer_maps_match` — Each prepared payload has a declared logical-to-physical map. Store, global layout, TMA image and each MMA reader compose to the same coordinates, including different instruction readers. Local writer/reader self-consistency alone is insufficient.
- Precondition `pre_publish` (implementation): `p.prep_completes_before_chain_and_all_workspace_readers_retire_before_reuse` — Prepared mode finishes every factor write and async store, then publishes through an ordered kernel/stream dependency before chain reads. Scratch is private to an invocation or its prior consumers have retired; this version does not use a cross-kernel flag ring.
- Precondition `pre_chain` (implementation): `p.unique_sequence_state_shards_and_ordered_chunk_consumption` — Each sequence/head/value shard has one state chain, consumes chunks in logical order, and finishes old-state projections before replacement. Value shards own disjoint state/output elements and share read-only factors.
- Precondition `pre_numeric` (implementation): `p.folded_factor_casts_and_state_operand_recipe_admitted` — Folding T into W/P and workspace stores change association/rounding. Preserve the explicit state-operand conversion used in X/output and the declared carried-state recipe; both supplies meet the admitted error policy rather than bitwise equivalence.
- Precondition `pre_backend` (implementation): `p.factor_supply_width_target_dtype_and_workspace_capacity_supported` — Selected on-chain or prepared route is implemented for this target/shape/dtype; actual chunk prefix fits capacity and producer/consumer use matching sequence origins. Dispatch occupancy thresholds are tuning policy, not legality or speed guarantees.
- Constraint `c_records` (implementation): `p.chunk_tokens == 16 and p.key_features == 128 and p.chunk_head_records >= 1 and p.factor_element_bytes == 2 and p.diagonal_element_bytes == 4` — First concrete binding:16-token,128-key factors, two-byte operand records and FP32 diagonal records. Counts describe logical useful chunk/head payloads.

## Local example and misuse

For a local instance of `map`, identify the object `parallel_feedback_factor_preparation_before_serial_state_chain` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/state_independent_chunk_factors@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/state_independent_chunk_factors@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: EK,EQ,W,P,E_end workspace; original values remain the chain inputs.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: prepared_logical_payload
  level: dram
  scope: kernel
  bytes: p.chunk_head_records * ((3 * p.chunk_tokens * p.key_features + p.chunk_tokens
    * p.chunk_tokens) * p.factor_element_bytes + p.key_features * p.diagonal_element_bytes)
  when: factor_supply == 'prepared_workspace'
  note: EK,EQ,W transposed records, P and E_end only. Allocation alignment, capacity
    slack, metadata, original V and state are separate.
overlap: []
```

## Obligations and numerical checks

- `independent_preparation` (proof): Every moved operation depends only on coherent chunk inputs, never a state generation or value residual.
- `factor_composition` (test): Folded W/P with original V preserves output and outgoing-state equations for each ordered chunk.
- `reader_coordinate_agreement` (test): Each native reader recovers the same labeled logical factor coordinates from the workspace image.
- `publication_and_lifetime` (test): All factors and metadata publish before chain consumers and all readers retire before workspace or old-state overwrite.

W/P factor folding introduces different rounding from an on-chain inverse/residual path. A workspace permutation valid for one reader can silently shift key channels for another.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- requires: chunked_delta_feedback@2 (outside menu); Supplies channelwise feedback factor equations and numerical boundaries; the schedule adds independent preparation and whole-workspace publication.

Structured contract: [template.yaml](template.yaml).
