# score_replication_or_exchange@1: two feature owners score supply

## When to consider

When two disjoint value-feature owners both need a complete probability tile, choose whether to recompute that tile in both owners or compute key halves once and exchange the weights. Replication spends score/normalization work to avoid P publication and shared reads. Sharding requires a common row reference, a complete P exchange and exactly one sum of mass shards. Both owners still read all keys; adding two already-complete replica denominators halves the output incorrectly.

## Computation before and after

Duplicate complete QK/P locally or partition QK by keys and exchange P, while output feature ownership remains fixed.

Keep owner j responsible for output features [j*Dv/2,(j+1)*Dv/2). With replicate_full_scores each owner computes the complete QK score tile including every latent/rope component, applies the same logical mask, forms its full online max and casts all exponent weights to T. Each replica accumulates the full rounded-weight mass and contracts all keys for its own value slice. No inter-owner P exchange or denominator sum is needed. With key_shards_exchange, each owner computes one key half, exchanges local maxima and obtains a common online row reference before exponentiating. Correct each old feature numerator and local mass to that frame. Cast local P to the same T, accumulate only that key half mass, and publish the local P half into its original row/key coordinates in shared memory. After publication each owner reads the entire shared P across both halves and contracts all keys for its value slice. At finalization sum the two mass shards once in their common frame; normalize each uniquely owned feature output by that full mass. The replicated path instead uses its already complete mass, never the sum of two complete replicas. Keep latent data alive through both score and value readers; an aliased rope/P region may be overwritten only after all rope readers retire. Empty rows use the declared nondividing branch.

Recover the original computation: Concatenate the two feature slices. In either path each output feature has the complete same-key weight/value sum; common-frame shard masses combine to the full mass already present in either replica.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| score_supply | compile / software | ["replicate_full_scores", "key_shards_exchange"] | Trade repeated complete score/normalization work for key-sharded score production plus common-frame P exchange. |

## Coupled constraints and scope

Two cooperative feature owners in one CTA, FP16/BF16 weights, FP32 numerator state and explicitly rounded-weight mass. Source classic MLA uses two 128-thread groups in a 256-thread CTA, 64 packed query rows, and shared latent values for score and PV. FP8 KV is first decoded and forces the replicated score path. Other owner counts, ungained high-precision mass definitions, probability gains and native asynchronous group protocols need separate bindings. Storage formulas describe the exchange regions, not total kernel allocation.

Scope predicate: `(p.same_query_key_values_masks_and_score_components_for_both_owners and p.complete_score_before_exponentiation) and (p.two_feature_slices_disjoint_complete_and_each_reads_all_keys) and (p.fp16_or_bf16_weight_cast_and_fp32_output_state and p.mass_uses_cast_weights_with_admitted_sum_order and p.redundant_score_paths_numerically_compatible) and (p.old_numerator_and_mass_corrected_before_new_terms and p.gain_absent_and_empty_row_state_defined) and (score_supply != 'key_shards_exchange' or (p.key_shards_disjoint_complete and p.common_max_published_before_exponents and p.p_exchange_exact_coordinate_map_and_generation and p.published_weights_visible_until_all_readers_retire and p.mass_shards_combined_once_in_common_reference)) and (score_supply != 'replicate_full_scores' or p.replicas_use_same_reference_and_complete_mass_without_cross_replica_sum) and (p.selected_score_supply_implemented_and_resources_fit and p.aliased_score_weight_latent_and_statistic_storage_lifetimes_valid)`.

- Precondition `pre_identity` (semantic): `p.same_query_key_values_masks_and_score_components_for_both_owners and p.complete_score_before_exponentiation` — Both owners use identical logical operands, masks and full score-component sums before normalization.
- Precondition `pre_output` (semantic): `p.two_feature_slices_disjoint_complete_and_each_reads_all_keys` — Output ownership is two disjoint feature halves, and both halves contract every logical key.
- Precondition `pre_numeric` (implementation): `p.fp16_or_bf16_weight_cast_and_fp32_output_state and p.mass_uses_cast_weights_with_admitted_sum_order and p.redundant_score_paths_numerically_compatible` — Both paths use the same declared P cast and rounded-weight mass definition; score/reduction differences satisfy the numerical policy.
- Precondition `pre_online` (semantic): `p.old_numerator_and_mass_corrected_before_new_terms and p.gain_absent_and_empty_row_state_defined` — Each online reference change corrects only old contributions; there is no hidden gain and empty rows have defined state/finalization.
- Precondition `pre_exchange` (implementation): `score_supply != 'key_shards_exchange' or (p.key_shards_disjoint_complete and p.common_max_published_before_exponents and p.p_exchange_exact_coordinate_map_and_generation and p.published_weights_visible_until_all_readers_retire and p.mass_shards_combined_once_in_common_reference)` — Sharded scores agree on one row frame, publish a complete P tile and sum the matching mass shards exactly once.
- Precondition `pre_replicas` (implementation): `score_supply != 'replicate_full_scores' or p.replicas_use_same_reference_and_complete_mass_without_cross_replica_sum` — Replicas each form the full row normalization and must not add identical complete masses together.
- Precondition `pre_backend` (implementation): `p.selected_score_supply_implemented_and_resources_fit and p.aliased_score_weight_latent_and_statistic_storage_lifetimes_valid` — The actual consumer implements the selected score layout, fits registers/shared capacity and retires every reader before rope/P, latent or max/mass storage reuse.
- Constraint `c_shape` (semantic): `p.owners == 2 and p.query_rows >= 1 and p.value_features >= 2 and p.value_features % 2 == 0 and p.key_tile >= 1` — Two owners have equal nonempty output-feature slices.
- Constraint `c_shard` (implementation): `score_supply != 'key_shards_exchange' or p.key_tile % 2 == 0` — Sharded score production uses equal nonempty key halves.
- Constraint `c_weight` (implementation): `p.weight_element_bytes == 2` — This binding exchanges FP16/BF16 probabilities.
- Constraint `c_capacity` (resource): `score_supply != 'key_shards_exchange' or 2 * p.query_rows * p.key_tile + 8 * p.query_rows <= p.exchange_capacity_bytes` — Weight tile and a two-owner FP32 row table fit the exchange allocation; max and final mass tables reuse the same bytes only after old readers retire.

## Local example and misuse

For a local instance of `contract`, identify the object `two_feature_owners_score_supply` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/score_replication_or_exchange@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/score_replication_or_exchange@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: Both feature owners need all weights; duplication avoids the exchange, while sharding avoids redundant score evaluation.. Storage: Key-shard exchange needs a full shared weight tile and one aliased two-owner max/mass table; source can also reuse retired rope storage.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: shared_weight_tile
  level: smem
  scope: cta
  bytes: 2 * p.query_rows * p.key_tile
  when: score_supply == 'key_shards_exchange'
- id: max_mass_tables
  level: smem
  scope: cta
  bytes: 8 * p.query_rows
  when: score_supply == 'key_shards_exchange'
overlap: []
```

## Obligations and numerical checks

- `score_and_output_identity` (proof): Full scores and masks match before exponentiation, and each unique output feature consumes every key once.
- `mass_multiplicity` (test): Sharded mass is combined once after frame alignment; replicated complete masses are never summed together.
- `exchange_and_alias` (test): P writes/read coordinates and generations agree; maxima publish before exponents, weights before PV, and all readers retire before rope/P or latent reuse.
- `numeric_and_resources` (test): The backend supports both selected score paths and their actual numeric, register/shared-memory and barrier contracts.

Rounded-P mass, shared max combination and different score/reduction association can change the result. Equal real-number weights alone do not prove equal native casts or replica states.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
