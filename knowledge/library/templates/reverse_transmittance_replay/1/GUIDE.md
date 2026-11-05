# reverse_transmittance_replay@1: reverse transmittance replay

## When to consider

When ordered alpha compositing needs gradients without saving every prefix, save the final transmittance and last accepted contributor, then reconstruct prefixes in reverse under the identical acceptance policy. Alphas0.2,0.5 give finalT0.4; reverse recovers prefixes0.8 then1.0. Including a forward-excluded saturating layer would divide by an extra1-alpha and corrupt all earlier gradients.

## Computation before and after

When ordered alpha compositing needs gradients without saving every prefix, save the final transmittance and last accepted contributor, then reconstruct prefixes in reverse under the identical acceptance policy.

Forward accumulates C+=T_before*alpha*c and T_after=T_before*(1-alpha), storing finalT and last accepted intersection. Reverse from that intersection with T_after=finalT and suffix_color=0. At layer i restore T_before=T_after/(1-alpha_i). With incoming color gradient gC and opacity-output gradient gA, compute dcolor_i=alpha_i*T_before*gC and dalpha_i=dot(T_before*c_i-suffix_color/(1-alpha_i),gC)+finalT/(1-alpha_i)*gA; background adds -finalT/(1-alpha_i)*dot(background,gC). Then suffix_color+=alpha_i*T_before*c_i and continue with T_after=T_before. Apply the local Gaussian/cap derivative branch and reduce/atomically accumulate primitive gradients. A forward-exclusive saturating contributor is omitted in reverse too.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

NVIDIA3DGS serial/sparse conic backward paths. MAX_ALPHA0.99 makes1-alpha>=0.01 while source denominator floor1e-6 is inactive. World-space saturation policies that store post-saturationT require separate checkpoint bindings; this version follows serial exclusive saturation. Resource effects: Saves finalT plus one last-ID per pixel rather than all per-layer prefixes; reverse recomputation loads/evaluates contributors again and reduces gradient traffic through warp sums.

Scope predicate: `p.forward_backward_order_and_predicates_identical and p.last_id_is_last_accepted and p.alpha_bounded_below_one and p.inverse_clamp_inactive and p.final_transmittance_and_background_match and p.gradient_accumulation_safe`.

- Precondition `pre_0` (semantic): `p.forward_backward_order_and_predicates_identical and p.last_id_is_last_accepted` — Reverse traversal reproduces the exact forward contributor order and skip/saturation decisions.
- Precondition `pre_1` (semantic): `p.alpha_bounded_below_one and p.inverse_clamp_inactive` — Every accepted alpha leaves positive1-alpha and any denominator floor is inactive or explicitly admitted.
- Precondition `pre_2` (semantic): `p.final_transmittance_and_background_match and p.gradient_accumulation_safe` — Saved finalT/background reflect the forward generation and shared primitive gradients combine without lost contributions.

## Local example and misuse

For a local instance of `map`, identify the object `reverse_transmittance_replay` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/gsplat/delta/reverse_transmittance_replay@1/template.yaml`. Original outline: `knowledge/studies/gsplat/delta/reverse_transmittance_replay@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Saves finalT plus one last-ID per pixel rather than all per-layer prefixes; reverse recomputation loads/evaluates contributors again and reduces gradient traffic through warp sums.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `prefix_recovery` (proof): Restored prefix matches the saved forward recurrence under admitted finite-precision inversion.
- `suffix_invariant` (proof): Suffix color equals contributions strictly after the current layer, scaled in the original forward frame.
- `acceptance_replay` (proof): Skipped/capped/last/saturating cases use the same branch and identity convention as forward.
- `gradient_ownership` (proof): Per-pixel contributions to shared primitive attributes are all retained through warp grouping and atomic accumulation.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
