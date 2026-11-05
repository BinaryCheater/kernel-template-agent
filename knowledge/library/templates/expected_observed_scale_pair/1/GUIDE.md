# expected_observed_scale_pair@1: expected observed scale pair

## When to consider

When scaling predictions and measured dynamic range are separate, retain the expected scale for current encoding and reduce an observed scale for later policy. Producer CTAs may atomically update observations, but current values must not divide by those partial updates. A future controller may choose another expected scale only after the whole observation domain completes.

## Computation before and after

Quantization consumes one immutable expected scale while separately reducing an observed output scale.

Load expected scale once for the invocation and quantize each computed value as value/expected_scale with the declared optional finite saturation/cast. Independently track absmax of the unscaled computed values, map NaN to the source exceptional statistic (+infinity), normalize that observed maximum by the format finite maximum and source epsilon, and atomically maximize the observed-scale destination. Never reinterpret a partial observed scale as the current encoding scale. Future expected scale choice is an external policy after producer completion; no history roll or same-invocation feedback is invented.

Recover the original computation: Decode current values with the immutable expected scale and expose a separate complete statistic about their unscaled source values.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Per-invocation expected scale plus independent observed max-statistic channel. Source flexpoint supports optional scale pointers; finite input subgroup suffices for ordinary numeric binding, exceptional policy is explicitly source-defined. This is not quantization_scale_state_transition@1 history roll/reset.

Scope predicate: `(p.expected_scale_positive_finite_and_immutable and p.expected_scale_matches_decode) and (p.observed_scale_initialized_and_domain_complete and p.observed_consumed_after_all_producers) and (p.exception_and_saturation_policy_explicit)`.

- Precondition `pre0` (semantic): `p.expected_scale_positive_finite_and_immutable and p.expected_scale_matches_decode` — Current encoded values and decoder use the same retained expected scale.
- Precondition `pre1` (semantic): `p.observed_scale_initialized_and_domain_complete and p.observed_consumed_after_all_producers` — Observed maximum has fresh state and is not consumed before all producing CTAs complete.
- Precondition `pre2` (semantic): `p.exception_and_saturation_policy_explicit` — NaN/Inf, epsilon, cast and optional saturation follow the requested recipe.

## Local example and misuse

For a local instance of `store`, identify the object `expected_observed_scale_pair` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/triton/delta/expected_observed_scale_pair@1/template.yaml`. Original outline: `knowledge/studies/triton/delta/expected_observed_scale_pair@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `epoch_separation` (proof): All values in this invocation use the same expected scale and observed updates cannot change their encoding.
- `statistic_complete` (test): Observed scale includes every valid source value once under the requested exception/max policy before consumption.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
