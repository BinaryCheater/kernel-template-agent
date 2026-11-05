# deferred_gradient_scalar@1: deferred gradient scalar

## When to consider

When saving a low-precision gradient that will be multiplied by a small global normalization, retain the unnormalized local derivative and apply the scalar after reload to avoid storing prematurely underflowed values.

## Computation before and after

When saving a low-precision gradient that will be multiplied by a small global normalization, retain the unnormalized local derivative and apply the scalar after reload to avoid storing prematurely underflowed values.

Compute local derivative g in accumulator precision, store q=cast_T(g), then in backward return cast_out(q*upstream/normalizer). Compare against store cast_T(g/normalizer): these are different rounded programs and must be explicitly admitted under the caller numerical policy. Keep the positive normalizer and upstream scalar associated with the same invocation; inactive entries store0. Source focal loss computes analytic derivative with focal weights then delays division by num_positives_sum to backward, where upstream loss gradient is also available. This trades later multiply/divide for a larger checkpoint magnitude, not exact commutation through rounding.

Recover the original computation: Scalar-normalized derivative checkpoints with source low-precision focal gradient; no bitwise equivalence to early normalization and no guarantee when unnormalized checkpoint overflows.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Scalar-normalized derivative checkpoints with source low-precision focal gradient; no bitwise equivalence to early normalization and no guarantee when unnormalized checkpoint overflows.

Scope predicate: `(p.scalar_normalizer_positive_finite and p.local_checkpoint_in_range and p.delayed_rounding_contract_admitted and p.invocation_scalar_matches_checkpoint)`.

- Precondition `pre0` (semantic): `p.scalar_normalizer_positive_finite and p.local_checkpoint_in_range and p.delayed_rounding_contract_admitted and p.invocation_scalar_matches_checkpoint` — Finite positive normalizer, no checkpoint overflow, admitted numerical policy and matching saved invocation.

## Local example and misuse

For a local instance of `map`, identify the object `deferred_gradient_scalar` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/apex/delta/deferred_gradient_scalar@1/template.yaml`. Original outline: `knowledge/studies/apex/delta/deferred_gradient_scalar@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `real_identity` (proof): Without rounding, moving one common scalar across storage preserves each derivative.
- `numeric_stage` (proof): Actual store/reload/output casts meet the admitted error/range policy, including near underflow.
- `scalar_lifetime` (proof): Every checkpoint uses its own normalizer and upstream scalar.

Floating-point reassociation, casts and approximate functions require an admitted numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
