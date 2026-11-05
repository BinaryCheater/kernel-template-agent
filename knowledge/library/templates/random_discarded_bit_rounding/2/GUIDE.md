# random_discarded_bit_rounding@2: random discarded bit rounding

## When to consider

When narrowing binary formats with stochastic carry, preserve the destination-specific exponent recipe and the exact random bits assigned to each element. FP16 needs13-bit carry plus a distinct subnormal grid; BF16 keeps16-bit truncation. Scalar mean, random correlations and bitwise replay are separate contracts.

## Computation before and after

When narrowing a binary format by discarding low significand bits, use a declared random-bit carry to select the adjacent representable value and keep exact per-element RNG identity across realizations.

For destination=bf16, retain version1: on the finite admitted range, reinterpret FP32 sign-magnitude bits, add uniform-or-declared r16 to the discarded low16 field, then truncate; use the per-element guarded special recipe outside that range. For destination=fp16 use13 random bits. Let b be absolute FP32 bits, E=b>>23, M=(b&0x7fffff)|0x800000. Source software forms z=b-(112<<23) for E>=113; z=M>>(113-E) for90<=E<113; z=0 for E<90. Add r13, set unsigned output magnitude=min((z+r13)>>13,0x7c00), then restore sign. Original Inf maps to Inf and original NaN to declared quiet NaN rather than perturbing classification. Normal finite nonoverflow inputs have away-from-zero probability equal to the13-bit residual/8192 if r13 is uniform. In FP16 subnormal range the software prepares a magnitude quantized down to units2^-37 before random carry; its mean equals that prepared value, with magnitude loss<2^-37, not arbitrary original FP32 x. Overflow has its declared Inf/saturation behavior and no finite unbiasedness guarantee. Packed FP16 low/high elements consume fields rbits[12:0]/[28:16] respectively. Preserve every element field/address/epoch for bitwise conversion replay; a different declared random law can preserve a scalar mean without preserving correlations or exact outputs.

Recover the original computation: Erase packing/generator realization and retain the exact declared random-carry conversion for each original FP32 bit pattern and random field. Uniform-field carry counts recover normal finite stochastic means; exponent rebasing and quantized subnormal preparation determine the separate FP16 law. The BF16 branch with discarded_bits16 recovers version1.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| discarded_bits | compile / boundary | [13, 16] | Discarded normal-significand bits, derived from destination:16 for BF16,13 for FP16. Not an independent accuracy setting. |
| destination | compile / boundary | ["bf16", "fp16"] | Bound output format. FP16 changes exponent/subnormal handling; it is not a direct16-bit BF16 truncation. |

## Coupled constraints and scope

Retains QuACK BF1616-bit carry input rule. Added FlashInfer FP16 software recipe from include/flashinfer/mamba/conversion.cuh:184-251 and GDN UMMA cvt.rs.f16x2 source operand/field mapping. Native UMMA supports the source enumerated SM100a/103a/107a .rs menu; fallback software is separately bound. Exact native subnormal/special equivalence remains an obligation. No generic FP8/FP4 scaling law, unconditional all-input unbiasedness, joint RNG independence or whole-kernel bitwise equivalence is inferred.

Scope predicate: `((destination == 'bf16' and discarded_bits == 16) or (destination == 'fp16' and discarded_bits == 13)) and ((destination == 'bf16' and p.bf16_finite_range_or_guarded_special_recipe) or (destination == 'fp16' and p.fp16_normal_subnormal_overflow_and_special_recipe_declared)) and (p.random_word_domain_joint_law_and_element_field_map_declared) and (p.narrowing_input_bits_and_all_prior_rounding_stages_fixed) and (p.low_high_operand_random_field_and_special_guards_match) and (p.selected_format_conversion_implemented_and_native_domain_match_proven)`.

- Precondition `pre_format` (implementation): `(destination == 'bf16' and p.bf16_finite_range_or_guarded_special_recipe) or (destination == 'fp16' and p.fp16_normal_subnormal_overflow_and_special_recipe_declared)` — BF16 retains the original finite-range/guarded-special recipe. FP16 uses explicit exponent rebasing, fixed-grid subnormal preparation and per-element Inf/NaN/overflow policy. Native/software substitutions must match the selected domain, not assume all exponents truncate identically.
- Precondition `pre_rng` (implementation): `p.random_word_domain_joint_law_and_element_field_map_declared` — Declare per-element random field, generator family/rounds, key/counter/pair component and invocation epoch. Uniform-field mean claims require uniformity under the relevant conditioning; joint independence and fresh-epoch claims are separate. Shared seed reuse or row LCG correlations are not erased by the scalar rounding rule.
- Precondition `pre_input` (implementation): `p.narrowing_input_bits_and_all_prior_rounding_stages_fixed` — Compare conversion realizations at identical FP32 input bits. A changed state accumulation or fold does not become bitwise equivalent merely by using the same random bits.
- Precondition `pre_pair` (implementation): `p.low_high_operand_random_field_and_special_guards_match` — Packed low/high elements use their own validity and random fields. Source FP16 low uses random bits12:0, high uses28:16, with unused bits ignored and PTX source operands deliberately reversed for low/high packing.
- Precondition `pre_backend` (implementation): `p.selected_format_conversion_implemented_and_native_domain_match_proven` — Supported target implements the chosen conversion. Native/software bit identity, especially FP16 subnormals/specials, is a separate obligation; source software tests do not establish GPU equality.
- Constraint `c_format` (implementation): `(destination == 'bf16' and discarded_bits == 16) or (destination == 'fp16' and discarded_bits == 13)` — Format fixes the normal mantissa carry width.

## Local example and misuse

For a local instance of `convert`, identify the object `random_discarded_bit_rounding` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/random_discarded_bit_rounding@2/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/random_discarded_bit_rounding@2/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `carry_threshold` (proof): A prepared residual d produces exactly d carries among2^discarded_bits uniform fields, including exponent carry boundaries.
- `fp16_exponent_recipe` (test): Normal exponent rebasing and subnormal fixed-grid preparation match the selected FP16 conversion; input values below2^-37 and finite overflow use their explicit laws.
- `special_guard` (test): Original per-element NaN/Inf/out-of-range classification cannot be changed by random perturbation or a sibling pair validity predicate.
- `rng_identity` (test): Logical element low/high fields and invocation identity remain stable under supported traversal/load regrouping when exact conversion replay is required.

FP16 subnormal preparation discards information below2^-37 in the source software law; overflow can create Inf. A finite-normal mean statement cannot justify arbitrary subnormals, reused RNG epochs, LCG joint independence or changed state-fold arithmetic.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
