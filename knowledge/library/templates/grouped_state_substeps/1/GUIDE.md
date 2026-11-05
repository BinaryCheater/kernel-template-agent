# grouped_state_substeps@1: real token to ordered state update coordinates

## When to consider

When a token performs several ordered state corrections, expose their update stream to chunked contractions without changing token boundaries. Flatten updates, apply decay only to the first, and gather outputs/checkpoints after the last. A64-update cut can lie inside a token, so preserve its carried state and gate phase.

## Computation before and after

Flatten ordered per-token updates while preserving real-token decay/readout/checkpoint boundaries.

For each sequence with L real tokens and n substeps, flatten m=n*t+j for0<=t<L,0<=j<n. Expanded inputs are K_m,V_m,beta_m from that exact substep. Set alpha_m=alpha_t for j0 and1 for j>0. token_serial executes these n updates before each token readout directly. expanded_chunks partitions m into64-update chunks, carries the actual state across every chunk (including within-token boundaries) and may use a correctly bound weighted feedback solve to compute residuals. Query/output t selects post-update position e_t=n*(t+1)-1, with causal mask m<=e_t; query t need not be materialized at the other positions. If queries are broadcast over several expanded chunks, accumulate each disjoint update contribution once and include the matching incoming-state contribution once. Store one output/checkpoint per real token. A real accepted prefix a commits exactly n*a updates. Tail/prefix coordinates stay sequence-relative; arbitrary chunk starts use their actual expanded phase. Pad only neutral updates and suppress their readouts/stores.

Recover the original computation: Group expanded updates n*t..n*t+n-1. Only the first applies alpha_t; ordered corrections match the original token update by induction. Selecting post-update n*(t+1)-1 recovers its inclusive output/checkpoint. Chunk partitioning only inserts intermediate carry boundaries; it cannot change the transition sequence.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| substeps_per_token | compile / boundary | [1, 2, 3, 4] | Workload-fixed ordered corrections per token. The concrete GDP decode menu admits1..4; changing this value changes the operator. |
| update_grouping | compile / software | ["token_serial", "expanded_chunks"] | Execute substeps under their token loop or flatten into64-update chunks with carried state and explicit final-substep output coordinates. |

## Coupled constraints and scope

GDP n1..4. Concrete lane-row decode n2 uses K128,V64, FP32 state and a64-thread CTA with four-lane state-row groups on its SM90/SM100/SM103 exported menu. SM90 V6 prefill walks64 expanded rows while retaining64-real-token Q groups and a rectangular causal mask. This rule is the semantic coordinate adapter; it does not claim source-identical query expansion, a generic native solve-width menu, approximate state resets or safe aliasing of arbitrary pooled checkpoints.

Scope predicate: `p.real_tokens >= 1 and p.expanded_chunk_updates == 64 and (p.ordered_delta_substeps_and_last_substep_readout_defined) and (p.real_and_expanded_sequence_prefixes_match_and_never_mix) and (p.decay_applied_only_at_first_substep_and_chunk_frame_phase_preserved) and (p.readout_and_checkpoint_boundaries_select_final_substep_only) and (p.expanded_chunk_boundaries_state_carry_and_neutral_tail_valid) and (p.normalization_gate_stages_and_substep_reassociation_admitted) and (p.state_snapshot_output_ownership_and_generation_lifetimes_valid) and (p.selected_grouping_substep_count_target_and_shape_supported)`.

- Precondition `pre_semantics` (implementation): `p.ordered_delta_substeps_and_last_substep_readout_defined` — Each real token first decays state once by its scalar alpha, then applies n ordered beta*(v-prediction) rank-one updates, then emits its query readout. n is part of workload semantics, not a tuning replacement.
- Precondition `pre_coordinates` (implementation): `p.real_and_expanded_sequence_prefixes_match_and_never_mix` — Real token t and substep j map to m=n*t+j relative to their sequence; global expanded prefix is n times the real prefix. K/V/beta are substep-indexed, Q/gate/output/checkpoint and accepted lengths are real-token-indexed.
- Precondition `pre_frames` (implementation): `p.decay_applied_only_at_first_substep_and_chunk_frame_phase_preserved` — Expanded decay alpha_m equals token alpha at j0 and1 otherwise. A chunk beginning inside a token does not repeat that token decay. Local cumulative frames subtract the complete prefix before the chunk, even if its boundary is within a token.
- Precondition `pre_readouts` (implementation): `p.readout_and_checkpoint_boundaries_select_final_substep_only` — Token t observes expanded post-update state at m=n*(t+1)-1. Its causal support includes exactly updates0..m. Output/checkpoint stores occur once per real token; accepted token count a maps to n*a committed updates.
- Precondition `pre_chunks` (implementation): `p.expanded_chunk_boundaries_state_carry_and_neutral_tail_valid` — Expanded chunks may bisect a token. The next chunk starts from the preceding update state, and every real readout uses all preceding substeps. Invalid padded updates have alpha1,beta0,key/value0 and cannot create stores.
- Precondition `pre_numeric` (implementation): `p.normalization_gate_stages_and_substep_reassociation_admitted` — Key/query normalization, beta/gate stages, prediction precision, rank-one update order and output/state casts have an explicit admitted numerical policy. Sequential and solve-based realizations do not inherit native bitwise equality.
- Precondition `pre_slots` (implementation): `p.state_snapshot_output_ownership_and_generation_lifetimes_valid` — Live state and checkpoint destinations have safe ownership; snapshots represent after-real-token states, old-state readers retire before overwrite, and input records match the same sequence generation.
- Precondition `pre_backend` (implementation): `p.selected_grouping_substep_count_target_and_shape_supported` — Selected token-serial or expanded-chunk route is implemented for n,target,shape and precision. Zero seeds, pooled initial state, frozen output and snapshot modes retain their separately supported menus.
- Constraint `c_domain` (implementation): `p.real_tokens >= 1 and p.expanded_chunk_updates == 64` — One nonempty sequence;64-update expanded prefill binding. Empty sequence identity copies are handled by the caller separately.

## Local example and misuse

For a local instance of `map`, identify the object `real_token_to_ordered_state_update_coordinates` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/grouped_state_substeps@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/grouped_state_substeps@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Expanded K/V/beta stream and original Q/gate/output coordinates, with intermediate chunk carry state.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `substep_induction` (proof): Flattening preserves every ordered update and exactly one decay per real token.
- `cross_chunk_phase` (test): Within-token chunk cuts preserve incoming state, gate-prefix origins and final-substep readout ownership.
- `prefix_units` (test): Real token prefixes, snapshots and accepted lengths select exactly the matching complete substep groups.
- `native_numeric_and_slots` (test): Implemented normalization/rounding and state-slot lifetimes satisfy the selected numerical and ownership contract.

Repeating the decay, skipping within-token feedback, using update counts as token counts, or reading out before the final correction changes the operator. Native gate log(alpha+epsilon), normalization and casts require a declared recipe.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
