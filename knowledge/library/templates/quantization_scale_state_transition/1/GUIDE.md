# quantization_scale_state_transition@1: quantization scale epoch

## When to consider

When quantization consumes a mutable per-tensor scale, make the producing data epoch and update barrier explicit. A requested current-tensor recipe reduces the whole tensor before casting; a requested delayed recipe consumes history and preserves its roll/reset invariant. These recipes change numerical values and are boundary choices, not freely interchangeable implementations.

## Computation before and after

Refine requested quantization into an epoch-specific maximum/scale/history state transition.

For current_tensor, reset amax, reduce absmax over every element of this logical tensor, then produce its FP32 multiplier with the requested epsilon and optional downward power-of-two rounding. A zero selected maximum returns multiplier 1 in the pinned current recipe. For a delayed recipe, snapshot history[0]; select max of the old window for history_max or that snapshot for history_recent. Produce multiplier scaled_max/selected_amax for positive maxima, clamping overflow to the requested finite cap; zero maximum retains the previous scale. Independently roll the old history left, wrap the snapshot to its last slot, and clear output slot 0 for the next producer. Capture reads before an in-place roll overwrites them. Only the matching scale epoch may quantize the consumer tensor; its reciprocal metadata must be paired with the scale actually consumed. Graph-noop behavior must preserve the recipe-specific state promised by its actual implementation.

Recover the original computation: Project the stateful implementation onto the requested quantization recipe: each cast consumes the specified epoch maximum and each history update has the specified next-producer slot.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| amax_policy | compile / boundary | ["current_tensor", "history_max", "history_recent"] | Requested source of the scale maximum. Current consumes this tensor; delayed uses the recorded history window or its current slot. |

## Coupled constraints and scope

Per-tensor FP32 multiplier recipes with finite nonnegative amax/history values and a positive finite previous scale. Source current recipe and delayed max/most-recent recipes have different zero and update policies; actual scale arithmetic, margin and epsilon are caller bindings. NaN/Inf amax propagation, cross-rank history aggregation and changes of scale granularity require separate bindings. No claim that delayed and current quantization are equal is made.

Scope predicate: `p.requested_policy_matches and (amax_policy == 'current_tensor' or p.history_length > 0) and p.scale_state_exclusive and p.history_snapshot_before_overwrite and p.consumer_epoch_matches and p.maximum_domain_complete and p.scale_update_visible_before_consumer and p.amax_values_finite_nonnegative and p.previous_scale_positive_finite and p.scale_arithmetic_policy_matches`.

- Precondition `pre_state` (semantic): `p.scale_state_exclusive and p.history_snapshot_before_overwrite and p.consumer_epoch_matches` — Scale/history ownership and all producing/consuming launches refer to the same tensor epoch.
- Precondition `pre_complete` (semantic): `p.maximum_domain_complete and p.scale_update_visible_before_consumer` — The selected tensor/window maximum is complete and the scale is visible before its consumer.
- Precondition `pre_values` (implementation): `p.amax_values_finite_nonnegative and p.previous_scale_positive_finite and p.scale_arithmetic_policy_matches` — The finite amax domain and previous scale satisfy the requested division, overflow and exceptional-zero policy.
- Constraint `c_recipe` (semantic): `p.requested_policy_matches` — Bind the requested quantization recipe rather than silently substituting another epoch.
- Constraint `c_history` (semantic): `amax_policy == 'current_tensor' or p.history_length > 0` — Delayed recipes require a nonempty history window.

## Local example and misuse

For a local instance of `contract`, identify the object `quantization_scale_epoch` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/transformerengine/delta/quantization_scale_state_transition@1/template.yaml`. Original outline: `knowledge/studies/transformerengine/delta/quantization_scale_state_transition@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One tensor multiplier and its optional amax history; reciprocal accompanies the consumed multiplier.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `epoch_binding` (test): Selected maximum, scale and output codes belong to the requested data epoch and complete domain.
- `history_transition` (proof): In-place roll preserves the old window needed for selection and clears only the next producer slot.
- `visibility` (test): Reset/reduction/update/consumer ordering and metadata ownership implement the stated state transition.

Changing the amax epoch or window changes the quantizer, saturation and error; zero behavior also differs between current and delayed recipes.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
