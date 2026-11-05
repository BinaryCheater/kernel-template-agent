# affine_activation_rematerialization@1: affine activation checkpoint boundary

## When to consider

When an MLP saves both affine input-to-activation and activation output for training, choose an earlier retention boundary to reduce long-lived storage. Save only the affine value and rerun activation, or retain original inputs and rerun affine then activation. Preserve every forward cast and weight generation. A ReLU AUX bitmask supports a derivative but cannot regenerate activation magnitudes; it cannot stand in for the full affine checkpoint in this rule.

## Computation before and after

Move the retained training boundary earlier across an affine-plus-activation chain and regenerate omitted values before backward consumers.

Let A=affine(X,W,b) and B=activation(A), with their forward stage precisions and casts fixed. Backward needs B for a downstream weight gradient and A for the activation derivative. With retention=affine_and_activation, save both A and B. With affine_only, save A and regenerate B=activation(A) before the downstream weight-gradient consumer. With affine_inputs, retain X,W,b and regenerate A then B before their consumers, using the matching forward logical operands, bias and stage precision. Retain the recomputed A until the activation derivative completes. All modes preserve other backward dependencies and residual/communication side effects. Do not rerun a stochastic/state-mutating producer without a separate replay contract or substitute a derivative-only AUX buffer for the full A value.

Recover the original computation: Replace every regenerated A/B with the corresponding saved forward value under the admitted numerical replay relation; the complete backward dataflow is recovered.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| retention | compile / software | ["affine_and_activation", "affine_only", "affine_inputs"] | Retained boundary of the deterministic affine-plus-activation chain; earlier boundaries trade more replay for fewer saved intermediates. |

## Coupled constraints and scope

Deterministic affine-plus-activation training chains with full-valued affine checkpoints and immutable replay operands. Source GELU and squared-ReLU MLP expose three retention levels. ReLU vendor AUX bitmasks, normalization-state reconstruction, arbitrary graph checkpointing and random/stateful replay need separate bindings. Boundary storage is explicit; it may already be retained for other gradient consumers and is not automatically a new allocation.

Scope predicate: `p.affine_and_activation_deterministic_and_side_effect_free and p.forward_stage_order_and_casts_declared and p.all_backward_value_and_derivative_consumers_enumerated and p.replay_inputs_weights_bias_and_auxiliary_state_unchanged and p.retained_state_coordinates_generations_and_lifetimes_match and p.retained_affine_checkpoint_is_full_value_not_derivative_only_codec and p.replayed_affine_activation_math_and_checkpoint_precision_admitted and p.replay_complete_before_all_backward_consumers`.

- Precondition `pre_chain` (semantic): `p.affine_and_activation_deterministic_and_side_effect_free and p.forward_stage_order_and_casts_declared and p.all_backward_value_and_derivative_consumers_enumerated` — The chain and all backward consumers have explicit semantics, including activation choice, bias placement and intermediate casts.
- Precondition `pre_inputs` (semantic): `p.replay_inputs_weights_bias_and_auxiliary_state_unchanged and p.retained_state_coordinates_generations_and_lifetimes_match` — Every replay sees the forward invocation's inputs, weights, bias and auxiliary state; retained values remain alive with their original identity.
- Precondition `pre_value` (semantic): `p.retained_affine_checkpoint_is_full_value_not_derivative_only_codec` — When an affine value is retained, it is sufficient to regenerate the activation output; a ReLU derivative bitmask alone is not such a value.
- Precondition `pre_numeric` (implementation): `p.replayed_affine_activation_math_and_checkpoint_precision_admitted and p.replay_complete_before_all_backward_consumers` — Replayed backend algorithms, rounding, activation approximation and casts satisfy the numerical contract, and each recreated value is ready before every consumer.

## Local example and misuse

For a local instance of `store`, identify the object `affine_activation_checkpoint_boundary` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flash-attention/delta/affine_activation_rematerialization@1/template.yaml`. Original outline: `knowledge/studies/flash-attention/delta/affine_activation_rematerialization@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: Retained forward inputs can regenerate intermediates needed by several backward stages.. Storage: Both intermediates, affine value only, or the earlier immutable boundary; transient replay storage is separate.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: both_values
  level: dram
  scope: kernel
  bytes: p.affine_bytes + p.activation_bytes
  when: retention == 'affine_and_activation'
- id: affine_value
  level: dram
  scope: kernel
  bytes: p.affine_bytes
  when: retention == 'affine_only'
- id: replay_boundary
  level: dram
  scope: kernel
  bytes: p.retained_boundary_bytes
  when: retention == 'affine_inputs'
overlap: []
```

## Obligations and numerical checks

- `backward_dependency_closure` (proof): Each retention mode supplies the full value or declared derivative state required by every backward consumer, with no omitted consumer.
- `immutable_replay_boundary` (test): Retained inputs, weights, bias and auxiliary data represent the same forward generation until replay finishes.
- `replay_numerics` (test): Replayed affine/activation and their casts/rounding satisfy forward-value and gradient accuracy policies for the selected backend.

A different GEMM heuristic, activation approximation, bias/cast placement or changed weight generation can alter replayed values and every dependent gradient.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
