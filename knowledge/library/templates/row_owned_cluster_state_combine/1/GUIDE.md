# row_owned_cluster_state_combine@1: row partitioned cluster raw attention merge

## When to consider

When a small attention tile is split across three clustered CTAs, distribute final rows among those CTAs instead of sending all outputs to one reducer. Retain local owned-row partials, publish other rows into DSM mailboxes and combine all peer raw states in one normalization frame. The eight-row split is 2/3/3 rows. Publish before remote reads and join again before reuse or peer exit; raw numerator weights differ from normalized-output merge weights.

## Computation before and after

Partition final rows across cluster peers and gather weighted raw state partials from distributed shared mailboxes.

For peer rank r in {0,1,2}, own rows [floor(8*r/3),floor(8*(r+1)/3)). Keep own-row raw numerator fragments in registers; publish nonowned row fragments to the row-packed shared mailbox, together with every row max and mass. Complete a cluster publication join. For each owned row find common max M across the three peers; assign a_s=exp2(m_s-M) only for positive-mass peers, otherwise zero. Let D=sum_s a_s*d_s and b_s=a_s/D when D>0, otherwise zero. Combine each owned feature as sum_s b_s*u_s using its local raw numerator and remote mailbox numerators, then cast/write the result once at the original query/head coordinates. Row-stat weights may be formed redundantly in peers, but output ownership remains disjoint. Do not multiply b_s by d_s again: u_s is unnormalized. If a slice is absent, predicate both publication and consumption consistently; padded output rows do not write. Finish a cluster reader-retirement join before mailbox reuse, alias overwrite or CTA exit.

Recover the original computation: Concatenate the three disjoint row ranges; each row equals the common-reference normalized sum of all original raw numerator/mass partials exactly once under the admitted numeric policy.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Three cooperating CTAs, eight packed query/head rows and 1024 value features with FP32 raw states. Source SM90 SWA M8S3 uses row-packed DSM and reuses retired KV storage. Two-peer natural-fragment mailboxes, global normalized partials, arbitrary cluster dimensions and different row/feature layouts require separate bindings; this version does not infer them from a generic DSM name.

Scope predicate: `p.key_partitions_disjoint_complete and p.raw_numerators_maxima_masses_match_same_query_and_support and p.nonempty_maxima_finite_and_masses_positive and p.empty_states_have_zero_mass_and_zero_finite_numerator and p.row_packed_mailbox_map_bijective_for_remote_features and p.output_map_covers_each_visible_row_feature_once and p.all_producer_mma_reads_retired_before_alias_mailbox_writes and p.mailbox_and_row_states_published_before_remote_reads and p.all_remote_readers_retire_before_reuse_or_cluster_exit and p.cluster_launch_and_dsm_peer_addresses_supported and p.all_peer_roles_reach_matching_cluster_joins_and_can_progress and p.common_reference_exponent_base_known_and_merge_rounding_admitted and p.local_raw_state_not_already_normalized`.

- Precondition `pre_states` (semantic): `p.key_partitions_disjoint_complete and p.raw_numerators_maxima_masses_match_same_query_and_support` — All peer states cover disjoint key partitions of the same rows, masks and score units.
- Precondition `pre_empty` (semantic): `p.nonempty_maxima_finite_and_masses_positive and p.empty_states_have_zero_mass_and_zero_finite_numerator` — Empty peers contribute zero without undefined exponential differences or nonfinite payload products.
- Precondition `pre_mapping` (implementation): `p.row_packed_mailbox_map_bijective_for_remote_features and p.output_map_covers_each_visible_row_feature_once` — Row packing preserves source/row/feature coordinates and final output ownership is exact.
- Precondition `pre_lifetime` (implementation): `p.all_producer_mma_reads_retired_before_alias_mailbox_writes and p.mailbox_and_row_states_published_before_remote_reads and p.all_remote_readers_retire_before_reuse_or_cluster_exit` — Dead operand storage may hold mailboxes only after old readers finish; both publication and terminal cluster join are required.
- Precondition `pre_cluster` (implementation): `p.cluster_launch_and_dsm_peer_addresses_supported and p.all_peer_roles_reach_matching_cluster_joins_and_can_progress` — The selected cluster supports remote shared reads, matching participation and forward progress.
- Precondition `pre_numeric` (semantic): `p.common_reference_exponent_base_known_and_merge_rounding_admitted and p.local_raw_state_not_already_normalized` — The merge consumes raw numerators with admitted FP32 weighting/reassociation and final cast.
- Constraint `c_binding` (implementation): `p.peers == 3 and p.rows == 8 and p.value_features == 1024 and p.partial_element_bytes == 4` — This first binding follows the three-peer eight-row FP32 source mailbox.
- Constraint `c_mailbox` (resource): `p.mailbox_capacity_bytes >= 4 * (8 * 1024 + 2 * 8)` — Each peer reserves row-packed numerator capacity plus max/mass slots in retired operand storage.

## Local example and misuse

For a local instance of `reduce`, identify the object `row_partitioned_cluster_raw_attention_merge` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/row_owned_cluster_state_combine@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/row_owned_cluster_state_combine@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: A local register partial avoids a self-mailbox store/read; other rows publish only for their remote final owner.. Storage: Per-peer row-packed FP32 mailbox and max/mass slots aliased with retired operand storage.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: row_mailbox
  level: smem
  scope: cta
  bytes: 4 * (8 * 1024 + 2 * 8)
overlap: []
```

## Obligations and numerical checks

- `row_and_feature_ownership` (proof): The half-open owner intervals partition eight rows, and each owned output feature reads all three same-coordinate raw contributions once.
- `raw_state_frame` (test): Every peer scale uses the same exponent/reference convention and the mass is applied only in the denominator, with defined all-empty behavior.
- `mailbox_mapping` (test): Remote row-packed writes and reads recover each original numerator feature without overlap or omission.
- `cluster_lifetime` (test): Publication follows all writes and retired MMA readers; terminal join follows all remote reads before storage reuse or peer exit.

FP32 FMA nesting and final BF16 conversion differ from a global normalized-partial combine; max/mass and raw numerator must remain from the same frame.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
