# empty_support_normalization@1: empty or positive exponential row

## When to consider

When masking or split attention can leave a row with no keys, handle that support boundary before normalization. Emit zero output and negative-infinite LSE without a sink; with a finite zero-valued sink a, emit zero and LSE=a. Include the sink once across splits, use the same weight gain as ordinary mass, and exclude NaN/underflow failures from the empty-support interpretation. Existing online_rescaled_reduction@2 requires positive final mass and cannot by itself admit the empty branch.

## Computation before and after

Guard normalization by semantic support and adopt a finite sink frame for otherwise empty rows.

Consume framed state (r,L,O,G) with G>0 and a reliable empty-support signal, implemented by L==0 under the range premise. For an empty row without sink, emit zero output and negative-infinite natural LSE without dividing by zero or forming r+log(0) with an undefined frame. For an empty row with finite sink a, adopt reference r=a, set L=G and O=0; emit zero output and LSE=a. For positive ordinary support retain its reference, optionally add G*exp(a-r) once to L, then emit O/L and r+log(L)-log(G), admitting all range and arithmetic effects. A sentinel stored maximum of negative infinity is metadata for empty support, never an operand in negative-infinity minus negative-infinity. Safe finite working references for masked prefixes must have produced zero weights/numerator before this final branch.

Recover the original computation: Expand the ordinary framed sums plus the optional zero-valued sink. Nonempty ratios recover the normalized weighted sum; empty support follows the explicitly defined zero-output and negative-infinite-LSE extension.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| sink | runtime / boundary | ["absent", "finite_logit"] | Whether the workload includes a finite zero-valued denominator-only contribution; not a tunable change to the attention contract. |

## Coupled constraints and scope

Masked/sparse exponential rows whose support may be empty, with absent or finite denominator-only sink and constant common positive gain. Sink overflow, finite weights underflowing to a false empty signal, nonfinite values, negative-infinite padded-head sinks and generic NaN repair are outside scope. Numerator/denominator casts and the output/LSE conventions must be admitted.

Scope predicate: `p.empty_state_has_zero_denominator_and_numerator and p.positive_state_common_frame_valid and p.zero_mass_iff_support_empty and p.valid_logits_and_values_finite and p.masked_logits_are_negative_infinity and p.no_nan_or_positive_infinity and p.sink_binding_matches_row and p.sink_owned_exactly_once and p.sink_and_ordinary_mass_share_gain and p.empty_output_zero_and_lse_negative_infinity_admitted and p.guarded_math_and_output_rounding_admitted`.

- Precondition `pre_state` (semantic): `p.empty_state_has_zero_denominator_and_numerator and p.positive_state_common_frame_valid and p.zero_mass_iff_support_empty` — The numerator is zero on an empty row; positive rows share one reference/gain; underflow or cancellation cannot masquerade as empty support.
- Precondition `pre_domain` (implementation): `p.valid_logits_and_values_finite and p.masked_logits_are_negative_infinity and p.no_nan_or_positive_infinity` — The branch handles semantic emptiness, not NaN repair or arbitrary infinite input.
- Precondition `pre_sink` (semantic): `p.sink_binding_matches_row and p.sink_owned_exactly_once and p.sink_and_ordinary_mass_share_gain` — Any finite sink belongs to this logical row once, including across split reductions, and its weight uses the ordinary common gain.
- Precondition `pre_contract` (semantic): `p.empty_output_zero_and_lse_negative_infinity_admitted and p.guarded_math_and_output_rounding_admitted` — The operator explicitly defines empty support as zero output and negative-infinite LSE, with guarded arithmetic and admitted output precision.

## Local example and misuse

For a local instance of `map`, identify the object `empty_or_positive_exponential_row` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flash-attention/delta/empty_support_normalization@1/template.yaml`. Original outline: `knowledge/studies/flash-attention/delta/empty_support_normalization@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Empty-support signal or a range-certified zero denominator.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `empty_detection` (test): The zero denominator branch identifies semantic empty support and its numerator is exactly zero under the admitted implementation.
- `guarded_arithmetic` (proof): Empty prefixes/final rows never require undefined infinity subtraction or zero division, and finite sink insertion establishes a valid common frame.
- `sink_ownership` (test): The finite sink is assigned to the correct logical row and included once across all split states.
- `nonempty_projection` (proof): Positive-state normalization and gain compensation preserve the existing framed reduction contract.

A zero or NaN machine denominator is not sufficient evidence of empty semantic support without the stated range/domain premises.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- alternative_of: [online_rescaled_reduction@2](../../online_rescaled_reduction/2/GUIDE.md); The empty final-row branch complements the positive-final-mass contract of version2; its positive projection uses that same common frame, without changing the streamed recurrence.

Structured contract: [template.yaml](template.yaml).
