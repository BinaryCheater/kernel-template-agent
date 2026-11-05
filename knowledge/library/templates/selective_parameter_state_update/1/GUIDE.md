# selective_parameter_state_update@1: activity gated parameter state

## When to consider

When sparse learned parameters are meant to stay untouched on inactive steps, gate their entire optimizer state transition, including moments and the local bias-correction clock. Source non-matrix Adam tests the incoming unscaled gradient before regularization. After one gradient2 update with beta=.5, moments(1,2) and clock1 remain unchanged on an inactive zero step; dense Adam would decay them to(.5,1) and advance to2. Only the declared freeze semantics permits that skipped work.

## Computation before and after

Gate a complete parameter state transition and its local clock as one unit.

Give each parameter an exclusive update owner. Evaluate its group's enable predicate and, for activity=enabled_and_nonzero, the incoming gradient zero test at the declared stage. Inactive owners return before writing weights, moments, regularization/decay or the local step counter. Active owners apply the original complete optimizer transition and increment the parameter clock once; use that clock for per-parameter moment debiasing. Publish the updated full-precision parameter and its admitted execution-precision copy together before their consumers. Preserve any independent global scheduler/wrapper clock and distinguish it from the local activity clock.

Recover the original computation: Expand the selected sparse optimizer map into active original transitions and inactive identity transitions. The state for every parameter and each independently retained global clock matches the declared coarse program.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| activity | compile / software | ["enabled", "enabled_and_nonzero"] | Within a declared parameter group, gate only on enablement or additionally on zero incoming gradient. |

## Coupled constraints and scope

Sparse per-parameter optimizers whose inactive semantics freeze moments, weights and local clocks. Does not justify skipping a dense zero-gradient update with moment decay, weight decay or global-clock bias correction. Source gates non-matrix Adam parameters before L2 regularization and clipping; matrix parameters may instead use enablement without the nonzero predicate.

Scope predicate: `p.inactive_transition_is_identity_for_complete_state and p.gate_uses_declared_gradient_stage and p.group_membership_consistent and p.gradient_values_finite and p.bias_correction_uses_parameter_active_clock and p.no_omitted_state_or_side_effects and p.parameter_state_exclusively_owned and p.gradient_epoch_ready and p.active_update_rounding_admitted`.

- Precondition `pre_identity` (semantic): `p.inactive_transition_is_identity_for_complete_state` — The requested optimizer semantics explicitly freeze every inactive parameter state component; dense zero-gradient Adam semantics do not satisfy this premise.
- Precondition `pre_gate` (semantic): `p.gate_uses_declared_gradient_stage and p.group_membership_consistent and p.gradient_values_finite` — Enable flags and gradient stage match the declared sparse activity contract; regularization or clipping cannot silently change the zero test.
- Precondition `pre_clock` (semantic): `p.bias_correction_uses_parameter_active_clock and p.no_omitted_state_or_side_effects` — Each active transition advances that parameter's clock once; global wrapper clocks are retained separately.
- Precondition `pre_owner` (implementation): `p.parameter_state_exclusively_owned and p.gradient_epoch_ready and p.active_update_rounding_admitted` — A single owner reads complete gradients and updates each parameter's full state with the admitted arithmetic.

## Local example and misuse

For a local instance of `map`, identify the object `activity_gated_parameter_state` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/tiny-cuda-nn/delta/selective_parameter_state_update@1/template.yaml`. Original outline: `knowledge/studies/tiny-cuda-nn/delta/selective_parameter_state_update@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: Inactive parameters avoid optimizer state traffic and arithmetic.. Storage: Optional local step counter per parameter in addition to optimizer moments.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `complete_state_gating` (proof): The inactive branch preserves all parameter-owned state, and the active branch executes its transition exactly once.
- `activity_clock` (proof): The local counter equals the number of admitted active transitions and supplies the matching bias correction.
- `update_publication` (test): Gradient readiness, state ownership and full/execution-precision parameter publication are correctly ordered.

Moving the zero test past regularization or using a global step in local debiasing changes sparse optimizer semantics.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
