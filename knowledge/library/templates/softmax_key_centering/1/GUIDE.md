# softmax_key_centering@1: common key translation before attention quantization

## When to consider

When key channels carry a large common offset before low-precision encoding, subtract one shared key mean across the entire row-normalization domain. The resulting query-key scores differ by one row constant, so ideal softmax output is unchanged and quantization sees a different range. Keep block scales independent of center scope. Restore the row constant for original logits/LSE, shift any sink consistently, and admit the actual centered quantization error rather than assuming equivalence.

## Computation before and after

Common key translation before quantization with optional row-offset restoration for non-normalized consumers.

With none, pass keys unchanged to the selected encoder. With channel_mean, sum valid key vectors per declared batch/head group and divide by the valid key count to obtain mu; exclude capacity padding from statistics. Replace each key K_j by K_j-mu before selecting quantization scales and encoding it. Keep quantization block scopes explicit: a common mu does not require one scale for the whole group. For query q and dot scale a the centered pre-quantization score is x_j-c with c=a*dot(q,mu), so softmax weights and the weighted value output need no c addition. Keep original masks/biases at their logical coordinates; if extra normalization logits exist, shift them by -c as well. If original-frame LSE or unnormalized scores are requested, add c to the centered result exactly once in the appropriate log units; LSE2 uses c*log2(e). Defined empty rows retain their output/LSE convention. Quantized centered keys and quantized original keys are different approximations: require an explicit error policy rather than treating centering as bitwise-transparent preprocessing. Recompute mu when input values change, even if preparation reuses addresses and shapes.

Recover the original computation: In exact arithmetic add c=a*dot(q,mu) to every centered row score and its centered LSE; row softmax and weighted values are unchanged. Additional subtraction/quantization errors are governed by the admitted numeric policy.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| centering | compile / software | ["none", "channel_mean"] | Leave keys unchanged or subtract the per-group mean key vector before encoding; selected backend must implement the choice. |

## Coupled constraints and scope

Forward attention with finite shared key centers and affine scaled dot logits followed by row softmax. Source dense noncausal SM90 head128 computes one FP32 K mean per batch/head, quantizes Q to INT8 per128-row block and centered K to INT8 per64-key half, and returns only BF16 output. Source has no sinks or returned LSE. The rule states necessary frame bindings for other callers but does not provide their native implementation. Nonlinear logit clipping, key-dependent centers without compensation, reused stale statistics and automatic backward/checkpoint semantics are excluded.

Scope predicate: `p.center_vector_finite_and_common_to_all_keys_of_each_normalized_row and p.finite_query_key_bias_and_row_offset and p.score_is_scaled_query_key_dot_plus_unchanged_bias_and_mask and p.query_operand_same_for_centered_score_and_row_offset and p.extra_normalization_logits_absent_or_shifted_by_same_row_offset and p.original_frame_lse_or_logits_absent_or_restored_with_row_offset and p.empty_row_convention_preserved and p.center_statistics_match_operand_generation_and_group and p.center_and_quantized_scales_published_before_readers_and_live_until_retirement and p.center_subtract_quantize_dot_and_softmax_error_policy_admitted and p.selected_centering_and_quantization_path_implemented_and_resources_fit`.

- Precondition `pre_center` (semantic): `p.center_vector_finite_and_common_to_all_keys_of_each_normalized_row` — Each normalized row sees one shared finite key-center vector, even if quantization scales use smaller blocks.
- Precondition `pre_scores` (semantic): `p.finite_query_key_bias_and_row_offset and p.score_is_scaled_query_key_dot_plus_unchanged_bias_and_mask and p.query_operand_same_for_centered_score_and_row_offset` — Finite operands, biases and row offsets admit an affine dot-product score; restoration uses exactly the selected query operand.
- Precondition `pre_extras` (semantic): `p.extra_normalization_logits_absent_or_shifted_by_same_row_offset` — A sink or other denominator-only logit must receive the same row shift, or the operator must have none.
- Precondition `pre_consumers` (semantic): `p.original_frame_lse_or_logits_absent_or_restored_with_row_offset and p.empty_row_convention_preserved` — Consumers needing original logits/LSE receive the row offset; output-only normalization may omit it.
- Precondition `pre_generation` (implementation): `p.center_statistics_match_operand_generation_and_group and p.center_and_quantized_scales_published_before_readers_and_live_until_retirement` — Center and code/scale generations agree; reusable prepared shapes do not imply reusable old value statistics.
- Precondition `pre_numeric` (implementation): `p.center_subtract_quantize_dot_and_softmax_error_policy_admitted and p.selected_centering_and_quantization_path_implemented_and_resources_fit` — Numerical policy includes cancellation, subtraction/casts, quantizer changes and selected contraction arithmetic.
- Constraint `c_domain` (semantic): `p.key_count >= 1 and p.head_features >= 1 and p.center_groups >= 1` — Mean statistics cover a nonempty valid key domain with one center per group.

## Local example and misuse

For a local instance of `contract`, identify the object `common_key_translation_before_attention_quantization` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/softmax_key_centering@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/softmax_key_centering@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: Subtract shared channel offsets to change the range presented to a low-precision key encoder without changing ideal row probabilities.. Storage: Shared FP32 key mean plus staged mean partials; quantized key buffers/scales are separate encoder storage.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: center_vectors
  level: dram
  scope: kernel
  bytes: 4 * p.center_groups * p.head_features
  when: centering == 'channel_mean'
- id: mean_partials
  level: dram
  scope: kernel
  bytes: 4 * p.center_groups * p.mean_partial_tiles * p.head_features
  when: centering == 'channel_mean'
overlap: []
```

## Obligations and numerical checks

- `row_shift` (proof): One center subtracts exactly one constant from every included score of a row, including any denominator-only terms.
- `frame_consumers` (test): Original logit/LSE consumers receive the matching offset in matching units; output-only normalization does not add it twice.
- `statistics_and_generations` (test): Valid-key counts, batch/head domains, padding and operand generations agree across mean, encoder and consumer.
- `quantized_numeric` (test): Centered subtraction and selected code/scale/contraction rounding meet the admitted approximation contract.

Centering may reduce or increase a quantization error depending on the data and scale policy. Finite-precision mean/subtraction, query encoding and contraction reassociation are not cancelled by ideal softmax shift invariance.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
