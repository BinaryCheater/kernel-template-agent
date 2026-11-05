# saturating_affine_round_away_quantization@1: saturating affine round away quantization

## When to consider

When scalar calibration defines integer endpoints, apply explicit endpoint saturation and half-away-from-zero affine rounding, preserving a matching inverse map. With min=-128,max127, s1,o0: x=-0.5 encodes-1 whereas nearest-even gives0; x0.5 encodes1. With equal endpoints a, x<=a maps-128 and x>a maps127, with s1 inverse metadata, not a constant decoder.

## Computation before and after

When scalar calibration defines integer endpoints, apply explicit endpoint saturation and half-away-from-zero affine rounding, preserving a matching inverse map.

Set qmin=-128,qmax=127. For max>min let s=(qmax-qmin)/(max-min), o=qmin-min*s; for equal endpoints set s=1,o=qmin-min. Encode x<=min as qmin, x>=max as qmax, otherwise lroundf(s*x+o) then signed int8 cast. Decode (q-o)/s. Endpoint branches take precedence, including the equal-endpoint case. Calibration may estimate endpoints from fixed-seed sorted row samples and quantile positions, but that policy is separate from this conversion.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| integer_bits | compile / software | [8] | Observed instantiated signed int8 format; numerical boundary choice. |

## Coupled constraints and scope

NVIDIA cuVS scalar quantizer instantiated for float/double/half to int8. Interior arithmetic uses the source temporary type and lroundf conversion, so cross-type precision changes need numerical review. Nonfinite and reversed calibration ranges excluded. Resource effects: Two affine metadata scalars, one conversion per input; saturation branches add comparisons, compact int8 storage handled separately.

Scope predicate: `p.finite_endpoints_and_inputs and p.min_not_above_max and p.half_away_rounding_requested and p.inverse_policy_matches`.

- Precondition `pre_0` (semantic): `p.finite_endpoints_and_inputs and p.min_not_above_max` — Calibration endpoints and inputs are finite and min<=max.
- Precondition `pre_1` (semantic): `p.half_away_rounding_requested and p.inverse_policy_matches` — Quantization contract specifies lroundf half-away rounding and inverse uses the identical affine metadata.

## Local example and misuse

For a local instance of `map`, identify the object `saturating_affine_round_away_quantization` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cuvs/delta/saturating_affine_round_away_quantization@1/template.yaml`. Original outline: `knowledge/studies/cuvs/delta/saturating_affine_round_away_quantization@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Two affine metadata scalars, one conversion per input; saturation branches add comparisons, compact int8 storage handled separately.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `rounding` (proof): Interior half values follow away-from-zero integer rounding, not nearest-even or truncation.
- `saturation` (proof): Endpoint branches prevent out-of-range integer conversion and preserve equality precedence.
- `metadata` (proof): Encoding and inverse use the same finite affine scale/offset and dtype conventions.
- `degenerate` (proof): Equal endpoints follow explicit two-endpoint step behavior rather than divide by zero.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
