# attention_value_axis_partition@1: score partition to value partition exchange

## When to consider

When value width makes each key owner's full output accumulator expensive, keep QK partitioned by keys but redistribute PV ownership across value features. First agree on row max/mass and exchange the complete weight tile; then each owner contracts all keys for only its feature slice. Four owners reduce each output numerator to one quarter at the cost of shared P and per-tile joins. Reading only the old key shard or mixing independently normalized P changes the result. Final output features need no cross-owner vector reduction.

## Computation before and after

Exchange key-sharded probabilities in a common online frame, then assign complete-key PV contractions to disjoint value-feature owners.

Initially each of four owners computes QK scores for its disjoint key shard. With full_value_partials, retain each owner's own online reference/mass and a full-feature weighted numerator, then merge all four states before output. With disjoint_value_features, combine the four row maxima with the prior common reference to obtain r_new, derive c=exp(r_old-r_new), and correct the old common mass. Each score owner computes its key weights exp(score-r_new), writes them once to shared P at their original row/key positions, and publishes its row weight sum. Combine sums into one common mass d. After publication, feature owner j consumes the complete shared P across all key shards and V[:,j*Dv/4:(j+1)*Dv/4], applying c to its old numerator before adding P*V. Repeat per key tile. Divide each disjoint feature numerator by the common d at finalization and write it once; no cross-owner output-vector combine is required. Preserve actual exponent base/score units, masked weights and defined V tails. Do not normalize each score shard independently and then concatenate its outputs, or let a feature owner read only its old key shard.

Recover the original computation: Concatenate the disjoint feature numerators and expand their complete key sums; this equals the merge of all key-sharded full-feature numerators in the same normalization frame under the admitted numerical policy.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| value_ownership | compile / software | ["full_value_partials", "disjoint_value_features"] | Keep one full value-output partial per key owner or exchange weights so each owner computes a disjoint feature slice. |

## Coupled constraints and scope

Four owners within one cooperative CTA query group, ordinary FP16/BF16 P staging and FP32 output accumulators. Source paged prefill selects feature ownership for value widths above256 divisible by the four owners, including width512. Other owner counts, FP8 P-gain paths, inter-CTA exchange or partial query/head layouts need additional implementation/numerical bindings. Resource expressions describe the numerator and exchange only, not total kernel usage.

Scope predicate: `p.participants == 4 and p.value_features % p.participants == 0 and p.key_tiles_disjoint_complete and p.query_rows_and_score_semantics_match_across_owners and p.value_features_have_unique_final_owners and p.same_defined_initial_state_and_empty_support_contract and p.old_numerator_corrected_before_new_weighted_terms and p.final_positive_mass_or_explicit_empty_branch and (value_ownership == 'full_value_partials' or (p.shared_weights_in_actual_common_reference and p.row_max_sum_publication_complete and p.probability_tile_exact_write_read_map and p.all_probability_readers_retire_before_reuse)) and (value_ownership == 'disjoint_value_features' or p.partial_outputs_merged_with_matching_normalizers) and p.fp16_or_bf16_weight_storage_with_fp32_accumulation and p.softmax_exp_weight_cast_and_reassociation_admitted and p.no_uncompensated_probability_gain and p.selected_organization_backend_legal_and_resources_fit`.

- Precondition `pre_partition` (semantic): `p.key_tiles_disjoint_complete and p.query_rows_and_score_semantics_match_across_owners and p.value_features_have_unique_final_owners` — Score shards cover each tile once and preserve query/head/mask identity; the feature-sharded path has one final writer per output feature.
- Precondition `pre_state` (semantic): `p.same_defined_initial_state_and_empty_support_contract and p.old_numerator_corrected_before_new_weighted_terms and p.final_positive_mass_or_explicit_empty_branch` — Both organizations preserve initialized online state, correct only old contributions and finalize zero-mass rows explicitly.
- Precondition `pre_exchange` (implementation): `value_ownership == 'full_value_partials' or (p.shared_weights_in_actual_common_reference and p.row_max_sum_publication_complete and p.probability_tile_exact_write_read_map and p.all_probability_readers_retire_before_reuse)` — Feature owners read every key shard's weight in one common reference, with completed row statistics and a safe complete probability-tile exchange.
- Precondition `pre_combine` (semantic): `value_ownership == 'disjoint_value_features' or p.partial_outputs_merged_with_matching_normalizers` — The full-feature partial path must combine both weighted numerators and their matching normalization states.
- Precondition `pre_numeric` (implementation): `p.fp16_or_bf16_weight_storage_with_fp32_accumulation and p.softmax_exp_weight_cast_and_reassociation_admitted and p.no_uncompensated_probability_gain` — This version binds ordinary FP16/BF16 probability staging and FP32 output accumulation; score/weight casts and changed association are admitted, with no unaccounted numerator gain.
- Precondition `pre_backend` (implementation): `p.selected_organization_backend_legal_and_resources_fit` — The selected organization has a supported layout/instruction implementation and fits actual register/shared-memory budgets; a source-derived dispatch choice is not automatically a free runtime knob.
- Constraint `c_shape` (semantic): `p.value_features >= 1 and p.query_rows >= 1 and p.key_tile >= 1 and p.weight_element_bytes == 2` — The represented row/key/feature tile is nonempty and the exchanged probability element is FP16/BF16.
- Constraint `c_participants` (implementation): `p.participants == 4 and p.value_features % p.participants == 0` — This source-backed binding uses four cooperating key/feature owners with equal output slices.
- Constraint `c_statistics` (resource): `p.row_statistics_bytes == 8 * p.participants * p.query_rows` — Each score owner publishes one FP32 max/sum pair per query row; this shared size is derived rather than independently tuned.

## Local example and misuse

For a local instance of `contract`, identify the object `score_partition_to_value_partition_exchange` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/attention_value_axis_partition@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/attention_value_axis_partition@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: The shared score/weight tile serves every feature owner without repeating QK.. Storage: Per-owner output numerator shrinks by four while shared P and per-tile row-statistic exchange are added.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: shared_weights
  level: smem
  scope: cta
  bytes: p.query_rows * p.key_tile * p.weight_element_bytes + p.row_statistics_bytes
  when: value_ownership == 'disjoint_value_features'
- id: feature_numerator
  level: reg
  scope: warp
  bytes: 4 * p.query_rows * (p.value_features // p.participants)
  when: value_ownership == 'disjoint_value_features'
- id: full_numerator
  level: reg
  scope: warp
  bytes: 4 * p.query_rows * p.value_features
  when: value_ownership == 'full_value_partials'
overlap: []
```

## Obligations and numerical checks

- `score_and_feature_partition` (proof): Every logical score and weighted output term appears once, and disjoint feature owners cover the original output without overlap or omission.
- `common_frame` (test): Every exchanged weight and output correction uses the same row reference/mass generation, including masked/empty transitions.
- `exchange_lifetime` (test): All P/statistic writes complete before readers, shared layouts agree, and every reader retires before reuse or alias overwrite.
- `numerical_binding` (test): Native exponent, probability conversion, MMA order and final division satisfy the declared numerical policy for each organization.

Common per-tile normalization and feature-owned PV change association and probability rounding relative to late merging of independently framed key-shard outputs.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
