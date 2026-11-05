# aligned_low_float_grid@1: aligned low float grid

## When to consider

When the target low-float conversion is unavailable but its finite magnitude grid embeds in another supported format, scale the subnormal knee into alignment and extract the target code from the carrier bits.

## Computation before and after

When the target low-float conversion is unavailable but its finite magnitude grid embeds in another supported format, scale the subnormal knee into alignment and extract the target code from the carrier bits.

For source E2M1 magnitudes0,0.5,1,1.5,2,3,4,6, multiply FP32 x by2^-14. FP16 magnitude patterns at target grid points become0x0000,0x0200,...,0x0e00. A toward-zero FP16 conversion supplies grid floor, then magnitude code=min((half_bits&0x7fff)>>9,7), plus original sign bit into nibble bit3. For stochastic rounding the source adds its declared scaled-grid random perturbation before floor; preserve that perturbation/RNG mapping as an input to this embedding, not an arbitrary random offset. Native RN E2M1 conversion can close an already exact target-grid value; no double-rounding equivalence is inferred for arbitrary carrier rounding.

Recover the original computation: Pinned E2M1-in-FP16 embedding with scale2^-14 and extraction bit9. Source rounding.py:727–750; native lowering, stochastic perturbation and special-value equivalence remain separate premises.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Pinned E2M1-in-FP16 embedding with scale2^-14 and extraction bit9. Source rounding.py:727–750; native lowering, stochastic perturbation and special-value equivalence remain separate premises.

Scope predicate: `(p.target_grid_equals_scaled_carrier_subgrid and p.carrier_floor_correct and p.saturation_special_policy_matches and p.perturbation_contract_admitted)`.

- Precondition `pre0` (semantic): `p.target_grid_equals_scaled_carrier_subgrid and p.carrier_floor_correct and p.saturation_special_policy_matches and p.perturbation_contract_admitted` — Grid embedding, toward-zero carrier conversion and exact exceptional-value/RNG contract.

## Local example and misuse

For a local instance of `convert`, identify the object `aligned_low_float_grid` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/quack/delta/aligned_low_float_grid@1/template.yaml`. Original outline: `knowledge/studies/quack/delta/aligned_low_float_grid@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `grid_identity` (proof): Each finite target code maps to the declared carrier bit pattern and extracts back to itself.
- `interval_identity` (proof): Carrier floor and bit extraction choose the target interval floor before saturation.
- `boundary_policy` (proof): Subnormal knee, saturation and NaN/sign conventions match the target conversion contract.

Floating-point reassociation, casts and approximate functions require an admitted numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
