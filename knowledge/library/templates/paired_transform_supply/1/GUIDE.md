# paired_transform_supply@1: paired rotation operand supply

## When to consider

When a rotation needs two feature components per output, choose how to obtain the original partner: co-own both in a vector, reload the other half, or exchange a matching register with a warp peer. Keep adjacent/split pairing and position coefficients fixed, preserve the inactive suffix and all required input casts, and prove original-pair reads precede any aliased write.

## Computation before and after

Supply complete original pairs to a coordinate-preserving two-component rotation map.

For each pair j in the active prefix R, define (a,b)=(2*j,2*j+1) for adjacent or(j,j+R/2) for split_halves. Capture original x[a],x[b] and the declared common pair coefficients c[j],s[j]. local_vector captures both in one thread; duplicate_load gives each output owner its own value and an additional original-partner load; warp_exchange supplies the peer register through the exact converged lane/slot map. Form y[a]=x[a]*c[j]-x[b]*s[j], y[b]=x[b]*c[j]+x[a]*s[j] in the declared native arithmetic and cast policy. Preserve x outside R. Keep any required norm/activation cast before the transform, and treat later scaling/quantization as separate refinements. In-place execution requires all original-pair readers, including other threads, to finish before destructive writes; locally loading a partner does not establish that global ordering.

Recover the original computation: Erase operand transport and evaluate the declared two-input transform at each original pair coordinate with the same checkpoints, coefficients and finite-precision order. Suffix coordinates map identically.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| pairing | compile / boundary | ["adjacent", "split_halves"] | Requested representation: pair(2j,2j+1) or(j,j+R/2) in the active prefix. Changing it changes the operation unless surrounding coordinates are remapped. |
| partner_supply | compile / software | ["local_vector", "duplicate_load", "warp_exchange"] | Choose the actual mechanism supplying the unchanged partner operand; legality follows the selected thread/fragment map and alias policy. |

## Coupled constraints and scope

FlashInfer pos_enc.cuh uses adjacent pairs within even local vectors or split-half partner loads. fused_qk_rmsnorm_rope.cuh uses split-half lane XOR16 for64/128/256-style head layouts with one head per warp and matching fragment slots. Pairing is a consumer contract; partner supply is an implementation choice. Native in-place load ordering, coefficient special values, quantized casts and cross-warp exchange need explicit bindings; no unconditional orthogonality or norm-preservation claim is made for rounded/scaled coefficients.

Scope predicate: `p.vector_width >= 2 and p.active_prefix >= 2 and p.active_prefix <= p.vector_width and p.active_prefix % 2 == 0 and (p.even_active_prefix_and_disjoint_complete_pair_map and p.untouched_suffix_preserved) and (p.position_frequency_and_pair_coefficients_explicit_and_same_generation) and (p.output_owner_and_partner_coordinate_maps_exact and p.selected_supply_implemented) and (partner_supply != 'local_vector' or p.both_pair_components_captured_by_same_thread) and (partner_supply != 'warp_exchange' or (p.partner_in_same_warp_and_fragment_slot and p.exchange_lanes_converged_and_mask_complete and p.native_shuffle_supported)) and (p.original_pair_reads_complete_before_overlapping_writes and p.source_destination_aliases_declared_safe) and (p.input_checkpoint_coefficient_math_fma_policy_and_output_cast_explicit and p.pair_transform_error_admitted) and (p.input_and_coefficients_published_before_reads and p.outputs_publish_before_dependent_consumers)`.

- Precondition `pre_pairs` (implementation): `p.even_active_prefix_and_disjoint_complete_pair_map and p.untouched_suffix_preserved` — Within each vector an even active prefix is partitioned into the requested adjacent or split-half pairs; the unrotated suffix keeps its original values.
- Precondition `pre_coefficients` (implementation): `p.position_frequency_and_pair_coefficients_explicit_and_same_generation` — Each pair has the requested coefficients and position/frequency recipe, whether computed or table-loaded. Position IDs are not inferred from output/cache addresses.
- Precondition `pre_layout` (implementation): `p.output_owner_and_partner_coordinate_maps_exact and p.selected_supply_implemented` — Each output owner receives its own and its original partner value under the selected actual vector/load/shuffle map.
- Precondition `pre_local` (implementation): `partner_supply != 'local_vector' or p.both_pair_components_captured_by_same_thread` — Local-vector mode captures both pair members before modifying either; pair boundaries do not straddle unrelated vector loads.
- Precondition `pre_exchange` (implementation): `partner_supply != 'warp_exchange' or (p.partner_in_same_warp_and_fragment_slot and p.exchange_lanes_converged_and_mask_complete and p.native_shuffle_supported)` — Lane exchange supplies the matching peer register with complete convergence/mask; cross-warp partners need an explicit different transport.
- Precondition `pre_alias` (implementation): `p.original_pair_reads_complete_before_overlapping_writes and p.source_destination_aliases_declared_safe` — All reads of an original pair precede any write that could destroy it, including redundant partner loads by another thread. Out-of-place output must not overlap other live input.
- Precondition `pre_numeric` (implementation): `p.input_checkpoint_coefficient_math_fma_policy_and_output_cast_explicit and p.pair_transform_error_admitted` — Preserve pre-transform BF16/FP16 checkpoints, coefficient approximation, FP32 multiply/add or FMA order, and final conversion under the numerical contract.
- Precondition `pre_publication` (implementation): `p.input_and_coefficients_published_before_reads and p.outputs_publish_before_dependent_consumers` — Pair values and coefficients are from one visible generation and complete outputs precede their downstream consumers.
- Constraint `c_shape` (semantic): `p.vector_width >= 2 and p.active_prefix >= 2 and p.active_prefix <= p.vector_width and p.active_prefix % 2 == 0` — Nonempty even pair domain inside each vector.

## Local example and misuse

For a local instance of `map`, identify the object `paired_rotation_operand_supply` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/paired_transform_supply@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/paired_transform_supply@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Thread-local pair registers, duplicate original loads or peer registers.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `pair_partition` (exhaustive): Each active coordinate has exactly one partner and output owner under the requested pairing; inactive suffix values are unchanged.
- `supply_identity` (test): Every local/load/shuffle partner is the original value at that coordinate and not a different pair or generation.
- `read_before_write` (test): All aliased pair readers finish before any destructive write; warp exchange masks and native ordering are valid.
- `numeric_checkpoint` (test): Coefficient law, pre-transform rounding, multiply/FMA order and output conversion match the requested transform.

Removing an input checkpoint, using the other pair convention, moving a nonuniform weight through the transform or reading an overwritten partner changes values. Coarser final FP8 quantization does not make an earlier BF16 rounding point redundant in general.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
