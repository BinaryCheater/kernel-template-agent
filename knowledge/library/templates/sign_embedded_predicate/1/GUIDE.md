# sign_embedded_predicate@1: nonnegative payload with sign predicate

## When to consider

When a nonnegative payload must travel with a Boolean mask, use its unused sign: kept v, dropped -v. Decode magnitude with abs and the predicate with signbit. A comparison-based consumer may use c>=0 only if the lost distinction between +0 and -0 cannot affect its result. Keep a signed copy until the last predicate user; clamping it first destroys the checkpoint.

## Computation before and after

Carry a Boolean in the sign of a finite nonnegative payload and decode it at each consumer.

For payload v>=0 and keep predicate m, encode magnitude abs(v) with a positive sign when m is true and a negative sign otherwise. Set the sign explicitly for zero as well, rather than inheriting an input negative zero. Store c instead of separate payload and predicate arrays. Decode magnitude as abs(c). Decode m as not signbit(c), or as c>=0 only under the zero-insensitivity premise. A kept-only payload consumer may use max(c,0) under the declared finite/zero semantics. Preserve the signed value for any later predicate consumer; creating a kept-only view does not replace it. An encoded payload is not an ordinary signed numerical intermediate until its predicate meaning has been decoded. Original signed/NaN payload domains require a different representation.

Recover the original computation: Recover the original magnitude and predicate from abs/signbit; in comparison mode recover observationally equivalent consumer results under the explicit zero-insensitivity premise.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| predicate_decode | compile / software | ["sign_bit", "nonnegative_comparison"] | Decode the predicate by sign bit, or by c>=0 only when zero-payload ambiguity cannot change any observed result. |

## Coupled constraints and scope

Finite nonnegative floating payloads with an available sign channel and consumers aware of the encoding. Source FA2 uses signed probabilities for dropout diagnostics and backward: ReLU conversion gives kept probabilities while signed FP32 values feed a centered derivative. This does not promise exact Boolean recovery from numerical comparison at zero, admit negative original payloads, or authorize moving the encoding across arbitrary math.

Scope predicate: `p.payload_finite_nonnegative and p.payload_magnitude_representation_admitted and p.sign_encoding_preserved_through_storage_and_transfers and p.payload_predicate_coordinates_and_generation_match and (predicate_decode == 'sign_bit' or p.all_consumers_insensitive_to_predicate_when_payload_zero) and p.every_consumer_decodes_sign_semantics_before_using_payload and p.no_unadmitted_sign_canonicalization`.

- Precondition `pre_payload` (semantic): `p.payload_finite_nonnegative and p.payload_magnitude_representation_admitted` — Payload values have no semantic negative sign or NaNs; any storage conversion of their magnitudes is numerically admitted.
- Precondition `pre_storage` (implementation): `p.sign_encoding_preserved_through_storage_and_transfers and p.payload_predicate_coordinates_and_generation_match` — The sign encoding survives every relevant transfer/cast, and predicate and magnitude belong to the same element/generation.
- Precondition `pre_zero` (semantic): `predicate_decode == 'sign_bit' or p.all_consumers_insensitive_to_predicate_when_payload_zero` — Numerical comparison cannot distinguish dropped -0 from kept +0, so it requires every consumer to be insensitive at zero.
- Precondition `pre_consumer` (semantic): `p.every_consumer_decodes_sign_semantics_before_using_payload and p.no_unadmitted_sign_canonicalization` — Consumers use the encoded sign deliberately; no unrelated activation, clamp, absolute value or arithmetic destroys it before the last predicate user.

## Local example and misuse

For a local instance of `store`, identify the object `nonnegative_payload_with_sign_predicate` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flash-attention/delta/sign_embedded_predicate@1/template.yaml`. Original outline: `knowledge/studies/flash-attention/delta/sign_embedded_predicate@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: One stored value carries both magnitude and predicate, with no separate predicate tensor.. Storage: One floating array, with its sign retained until all predicate consumers finish.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: encoded_payload
  level: dram
  scope: kernel
  bytes: p.elements * p.payload_element_bytes
overlap: []
```

## Obligations and numerical checks

- `encoding_and_zero` (test): Positive payloads recover magnitude/predicate exactly; zero uses preserved sign-bit decoding or a verified zero-insensitive consumer contract.
- `representation_lifetime` (test): Storage conversion and transfer preserve the required sign until all predicate consumers finish, with matching coordinates/generations.
- `consumer_equivalence` (test): Every consumer decodes the sign before use, and its magnitude rounding/zero behavior satisfies the original computation.

Casting a small negative encoded value to negative zero loses predicate information under c>=0 even if its sign bit survives; zero-insensitivity must cover every downstream use.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
