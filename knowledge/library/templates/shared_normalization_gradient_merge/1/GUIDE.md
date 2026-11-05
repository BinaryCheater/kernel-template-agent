# shared_normalization_gradient_merge@1: shared normalized row gradient fan in

## When to consider

When a normalized input row is shared across several consumers, choose where to hold their FP32 gradient sum, then apply one shared normalization pullback using raw inputs and saved statistics before the final narrow cast. Preserve consumer order separately from the complete-channel dot tree.

## Computation before and after

Sum gradients from all consumers of one normalized row before its shared normalization pullback.

For each shared raw row, enumerate its consumers in the fixed group order. With owned_accumulator, initialize D from consumer0 and add each later FP32 D_j into the unique owner accumulator. With fp32_partials, store each D_j in its exclusive row/consumer/channel slot, publish all required slots and perform the same ascending FP32 additions later. Use original raw x and saved forward r to form unrounded FP32 y=x*r. Compute the complete-row dot s=dot(D,y) with the declared channel tree, then dx=r*(D-y*s) with explicit product/subtract/scaling stages. Round dx once into the output format. Do not substitute forward narrow cast(x*r) for y or independently cast/pull back consumers before merging. Preserve raw input, saved statistic, normalized operand and output coordinate/stride identities; tail tokens and empty domains do not create consumers.

Recover the original computation: Each raw row is shared, so its incoming smooth gradient is the sum of exactly its consumer gradients. Apply the shared normalization Jacobian J=r*I-r^3*x*x^T once to that sum, respecting the fixed finite-precision realization. Original raw/statistic generation identifies the same Jacobian used by the source backward convention.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| gradient_merge_storage | compile / software | ["owned_accumulator", "fp32_partials"] | Keep the ordered consumer sum in one group owner or materialize exact per-consumer FP32 slots for a later ordered merge. Pullback and final-cast stages stay the same. |

## Coupled constraints and scope

FlashInfer deterministic KCP GDN SM100/103/107, BF16 raw/forward/output rows, 128 channels and HV/Hqk contiguous consumers. Forward r is FP32 inverse sqrt of complete sum of squares plus1e-6. Native fused mode uses a persistent chunk/QK-head owner and TMEM ordered group sum; host-orchestrated mode writes per-value-head FP32 partials and reduces token blocks before the same pullback. This defines the smooth normalization gradient convention through the narrow activation stage; it does not differentiate the discrete BF16 quantizer or promise bitwise equality under alternate trees.

Scope predicate: `(p.consumer_count >= 1 and p.channels >= 1 and p.live_rows >= 0) and (p.one_raw_normalized_row_shared_by_exact_consumer_group) and (p.forward_rounding_and_normalization_gradient_convention_explicit) and (p.raw_row_saved_r_and_local_gradients_share_one_forward_generation) and (p.complete_fp32_consumer_gradients_merge_in_fixed_group_order) and (p.complete_row_dot_tree_and_fp32_operation_recipe_explicit) and (gradient_merge_storage != 'fp32_partials' or p.partial_slots_publish_before_read_and_live_through_final_merge) and (gradient_merge_storage != 'owned_accumulator' or p.one_owner_holds_group_accumulator_through_pullback) and (p.output_strides_tail_masks_and_unique_raw_gradient_owner_valid) and (p.selected_target_group_ratio_and_norm_programs_implemented)`.

- Precondition `pre_shared` (semantic): `p.one_raw_normalized_row_shared_by_exact_consumer_group` — One raw row x is normalized once and shared by exactly the enumerated consumer group. Each consumer supplies its complete gradient D_j with respect to that same normalized row; no consumer is omitted or counted twice.
- Precondition `pre_stage` (semantic): `p.forward_rounding_and_normalization_gradient_convention_explicit` — Forward supplies cast_narrow(x*r), with saved forward r. Backward convention applies the smooth normalization pullback r*(D-y*dot(D,y)) with y=x*r at the declared FP32 stages, not the derivative of a discrete quantizer. Epsilon and statistic domain match forward.
- Precondition `pre_generation` (implementation): `p.raw_row_saved_r_and_local_gradients_share_one_forward_generation` — Raw row, saved r, consumer gradients and head-group map refer to one forward generation; normalized narrow activations cannot reconstruct the unrounded y. Replayed r is interchangeable only with an explicit equality contract.
- Precondition `pre_order` (implementation): `p.complete_fp32_consumer_gradients_merge_in_fixed_group_order` — Initialize from the first gradient then add subsequent consumer IDs in fixed ascending order in FP32. No per-consumer narrow cast, atomics or schedule-dependent reassociation occurs before the shared pullback. Singleton groups bypass addition without changing the final stages.
- Precondition `pre_dot` (implementation): `p.complete_row_dot_tree_and_fp32_operation_recipe_explicit` — The complete normalized channel row participates in dot(D,y). Dot tree, product/add/FMA grouping, subtraction, final scaling, FTZ and final cast follow the admitted recipe. Group-sum order and channel-dot order are different contracts.
- Precondition `pre_workspace` (implementation): `gradient_merge_storage != 'fp32_partials' or p.partial_slots_publish_before_read_and_live_through_final_merge` — Materialized per-consumer partials have exact token/group/consumer/channel coordinates and nonoverlapping writers. Reducers read only published live slots; buffers remain immutable until all required rows are reduced.
- Precondition `pre_owned` (implementation): `gradient_merge_storage != 'owned_accumulator' or p.one_owner_holds_group_accumulator_through_pullback` — A unique group owner serially merges all consumer contributions in its FP32 accumulator and retains it through the complete-row pullback; accumulator capacity and internal barriers are supported.
- Precondition `pre_output` (implementation): `p.output_strides_tail_masks_and_unique_raw_gradient_owner_valid` — One owner writes each raw gradient element after the pullback and one final narrow cast. Raw/normalized/output strides are distinct, chunk tails are masked and packed sibling gradients are disjoint.
- Precondition `pre_backend` (implementation): `p.selected_target_group_ratio_and_norm_programs_implemented` — The selected backend implements normalized preparation, saved-statistic application and raw pullback for the shape and head ratio. Source host orchestration fallback still executes tensor operations on the GPU; it is not CPU numerical execution.
- Constraint `c_shape` (semantic): `p.consumer_count >= 1 and p.channels >= 1 and p.live_rows >= 0` — Every live shared row has at least one consumer and complete positive channel extent; empty token domains write nothing.

## Local example and misuse

For a local instance of `reduce`, identify the object `shared_normalized_row_gradient_fan_in` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/shared_normalization_gradient_merge@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/shared_normalization_gradient_merge@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Group-owned FP32 accumulator or per-consumer FP32 materialization; raw x and saved forward r survive through pullback.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: materialized_consumer_gradients
  level: dram
  scope: kernel
  bytes: p.live_rows * p.consumer_count * p.channels * 4
  when: gradient_merge_storage == 'fp32_partials'
  note: One FP32 consumer-gradient tensor. Q and K have separate instances. Native
    fused path keeps its accumulators in TMEM; this effect counts only materialized
    partials.
overlap: []
```

## Obligations and numerical checks

- `consumer_fan_in` (test): Every consumer gradient contributes exactly once to its shared raw row in the prescribed FP32 order.
- `pullback_stage` (proof): The admitted raw/statistic Jacobian and complete-row dot are applied after fan-in and before the sole final cast.
- `storage_ownership` (test): Owned accumulators or published FP32 partial slots supply identical admitted inputs to the pullback and remain valid through their last reader.
- `numeric_recipe` (test): Forward/statistic reconstruction and backward FP32 operation trees, no-FTZ behavior and final rounding match the native stage contract.

Casting per-head gradients early, substituting BF16 normalized rows for raw x*r, or changing either group addition or channel-dot order changes the raw gradient. Saved-r equality is a stage identity, not a consequence of matching shapes.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
