# carry_compensated_parameter_split@1: split master parameter bits

## When to consider

When a mixed-precision optimizer retains both an FP32 master and a BF16 model view, share the master high word with the model and keep only the low word as a remainder. Undo the signed-low carry before reconstructing the FP32 update value. This saves the separate master allocation only if initialization, rounding and every update use the same paired bit codec.

## Computation before and after

Lossless FP32 master bit reconstruction from a carry-adjusted BF16 model word and signed low word.

View finite FP32 master bits as high word H and low word L. Store the BF16 model bit word P=H+c where c is one exactly when L interpreted as signed int16 is negative (its top bit is set); store L unchanged as the sixteen-bit remainder. To reconstruct, subtract c from P and concatenate that corrected high word with the unchanged low word. Run the optimizer update on the reconstructed FP32 value, then split its result with the same codec. The remainder is raw low bits, not an arithmetic residual that may be added numerically. This source uses an increment at exact midpoint L=0x8000, including when H is even; it is not nearest-even BF16 conversion.

Recover the original computation: Undo the implicit high-word carry and concatenate the retained low word; the exact admitted FP32 master bit pattern is recovered.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Finite FP32 master and BF16 model parameters with signed sixteen-bit low-word remainder and carry-aware initialization. Source AdamFunctorMasterParamRemainder in common/multi_tensor/adam.cu reconstructs and updates this pair. High-word arithmetic must remain representable and rounded model finite. Arbitrary BF16 nearest-even initialization, FP16 model formats, NaN/Inf handling and a float-valued residual representation require another binding.

Scope predicate: `p.fp32_master_bf16_model and p.word_order_matches and p.signed_low_word_semantics_match and p.initial_pair_uses_same_codec and p.model_rounding_policy_admitted and p.high_word_arithmetic_in_range and p.model_and_remainder_paired and p.optimizer_has_exclusive_update and p.no_partial_pair_visible and p.master_and_rounded_model_finite and p.fp32_update_arithmetic_preserved`.

- Precondition `pre_format` (implementation): `p.fp32_master_bf16_model and p.word_order_matches and p.signed_low_word_semantics_match` — The target exposes the FP32 high/low sixteen-bit words with the expected BF16 and signed-low interpretation.
- Precondition `pre_codec` (semantic): `p.initial_pair_uses_same_codec and p.model_rounding_policy_admitted and p.high_word_arithmetic_in_range` — Initial and updated pairs use this carry codec; the model-view rounding and signed-word arithmetic are admitted.
- Precondition `pre_ownership` (implementation): `p.model_and_remainder_paired and p.optimizer_has_exclusive_update and p.no_partial_pair_visible` — A parameter and its low word remain paired and readers see a completed update.
- Precondition `pre_values` (semantic): `p.master_and_rounded_model_finite and p.fp32_update_arithmetic_preserved` — The finite-domain master update retains its requested FP32 arithmetic before splitting.

## Local example and misuse

For a local instance of `store`, identify the object `split_master_parameter_bits` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/transformerengine/delta/carry_compensated_parameter_split@1/template.yaml`. Original outline: `knowledge/studies/transformerengine/delta/carry_compensated_parameter_split@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Four bytes total for model plus remainder instead of BF16 model plus a separate four-byte master.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `bit_identity` (proof): Subtracting the carry and concatenating L recovers every admitted master word exactly.
- `view_rounding` (test): The model BF16 view and initialization obey the required carry/tie codec rather than a different rounding rule.
- `pair_visibility` (test): Model/remainder ownership and publication preserve the pair across optimizer updates.

Master reconstruction is bit-exact within the codec; the model view follows a source-specific midpoint increment and is not automatically interchangeable with nearest-even BF16.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
