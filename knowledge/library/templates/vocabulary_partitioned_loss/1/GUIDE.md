# vocabulary_partitioned_loss@1: partitioned logit loss statistics

## When to consider

When class logits are distributed by vocabulary, keep them sharded and communicate only each row's local log-normalizer and additive target/smoothing contribution. Merge log-normalizers with logsumexp, sum additive terms, then add the global normalizer and squared-LSE penalty once. Save that global LSE for every local backward slice; the target term belongs to exactly one shard.

## Computation before and after

Combine local log-normalizers and additive target/smoothing terms, then differentiate each disjoint vocabulary slice using the final global normalizer.

For scaled logits z=a*x, class count C, smoothing epsilon, label y and squared-LSE coefficient lambda, shard r computes local L_r=logsumexp(z_r) and A_r=-(epsilon/C)*sum(z_r)-(1-epsilon)*z_y if it owns y, retaining only the smoothing term otherwise. Merge L=logsumexp_r(L_r) and A=sum_r(A_r). Final row loss is L+A+lambda*L*L; add L and its squared penalty once, not once per shard. Ignore-index rows produce loss and gradient zero. Save final global L for backward. Each shard reconstructs P_c=exp(z_c-L) and writes g*a*((1+2*lambda*L)*P_c-epsilon/C-(1-epsilon)*[c=y]) for its owned classes. Local class indices are rebased to global labels using the immutable shard start/map. Only row scalars need communication; logits stay sharded. Any separate z-loss diagnostic is not an extra differentiable loss output unless the caller explicitly defines that contract.

Recover the original computation: Concatenate disjoint class shards and expand the local additive terms; nested logsumexp equals full-vocabulary logsumexp, recovering the same smoothed cross entropy, squared-LSE penalty and local gradient slices.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Fixed complete vocabulary partitions and finite row normalization, including label smoothing and squared natural-LSE penalty. Source uses equal contiguous shards C=world_size*local_columns, gathers local LSE scalars and sums additive losses. Unequal/noncontiguous shards need explicit map/count bindings. Empty shards, invalid nonignored labels, global-LSE-as-local substitution and changing distributed ownership during backward are outside this version.

Scope predicate: `p.vocabulary_shards_disjoint_complete_and_nonempty and p.global_class_count_and_target_identity_match and p.each_nonignored_target_has_one_owner and p.finite_scaled_logits_and_positive_finite_mass and p.score_scale_positive_and_same_across_shards and p.smoothing_and_zloss_parameters_match and p.local_statistics_cover_actual_shard and p.collective_row_rank_maps_and_visibility_valid and p.global_normalizer_added_once and p.backward_uses_final_global_lse_and_original_local_logits and p.ignored_rows_zero_loss_and_gradient and p.local_gradient_output_ownership_valid and p.logsumexp_collective_reduction_and_gradient_casts_admitted`.

- Precondition `pre_shards` (semantic): `p.vocabulary_shards_disjoint_complete_and_nonempty and p.global_class_count_and_target_identity_match and p.each_nonignored_target_has_one_owner` — Every class belongs to one shard and every nonignored target has exactly one owner; global class count and row identities agree.
- Precondition `pre_scores` (semantic): `p.finite_scaled_logits_and_positive_finite_mass and p.score_scale_positive_and_same_across_shards and p.smoothing_and_zloss_parameters_match` — Use the same positive score scale, smoothing epsilon and squared-LSE coefficient on every shard, with finite logits/mass.
- Precondition `pre_statistics` (implementation): `p.local_statistics_cover_actual_shard and p.collective_row_rank_maps_and_visibility_valid and p.global_normalizer_added_once` — Local or precomputed LSE covers only the actual shard; collectives preserve row/rank identity, and the final global LSE/z-loss contribution has one logical addition per output.
- Precondition `pre_gradient` (semantic): `p.backward_uses_final_global_lse_and_original_local_logits and p.ignored_rows_zero_loss_and_gradient and p.local_gradient_output_ownership_valid` — Backward consumes the final global checkpoint and owns disjoint local logit gradients, with ignored rows zeroed.
- Precondition `pre_numeric` (implementation): `p.logsumexp_collective_reduction_and_gradient_casts_admitted` — Local exponential statistics, scalar collectives and gradient conversion satisfy the numerical contract.

## Local example and misuse

For a local instance of `reduce`, identify the object `partitioned_logit_loss_statistics` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flash-attention/delta/vocabulary_partitioned_loss@1/template.yaml`. Original outline: `knowledge/studies/flash-attention/delta/vocabulary_partitioned_loss@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: One final row LSE serves all local class-gradient blocks without gathering logits.. Storage: Local logits plus row scalar statistics; communication is per row/shard rather than per class logit.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `shard_statistics` (proof): Class shards form the complete vocabulary once, each local statistic uses that shard, and the target contributes at exactly its owning shard.
- `global_loss_and_checkpoint` (test): Collectives combine matching row scalars and add the final global normalizer and z-loss once, publishing global L before backward.
- `gradient_and_ignore` (test): Each local output equals its full-vocabulary gradient slice with complete smoothing/scale/z-loss terms and zero ignored rows.
- `numeric_collectives` (test): Actual local reductions, collective precision/order and gradient casts meet the numerical policy.

Logsumexp nesting and collective order change rounding; using a local normalizer in backward or duplicating an already global LSE changes the mathematical probabilities.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
