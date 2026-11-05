# rounding_residual_checkpoint@1: paired arithmetic rounding checkpoint

## When to consider

When a low-precision output is also a training checkpoint and its rounding corrupts a later statistic, optionally save the rounding residual in a second low-precision array. Reconstruct both in the accumulator format before use. BF16 high1 plus low2^-9 recovers FP32(1+2^-9), but more detailed residuals can still round away. The pair improves a numerical checkpoint under its admitted error policy; it is not lossless storage or necessarily fewer bytes.

## Computation before and after

Save a rounded value and optionally its rounded arithmetic residual; reconstruct before consuming.

Let A be the producer accumulator format, T the smaller checkpoint format, and x the already-finalized A value, including its final output scale. Set h=round_T(x). For rounded_only save h and reconstruct xhat=A(h). For rounded_and_residual compute r=fl_A(x-A(h)), save l=round_T(r), and reconstruct xhat=fl_A(A(h)+A(l)). Save h and l with identical coordinate/generation ownership. The residual is computed from the same x just rounded into h, before releasing that producer fragment. A consumer may stream chunks, reconstruct each chunk, and retain only its reduction partials. Do not add h+l in T, compute the residual after discarding x, or treat the pair as an exact bit codec.

Recover the original computation: Decode the checkpoint with its selected encoding to the admitted approximation xhat of the original finalized accumulator value x, retaining its logical coordinate and generation.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| encoding | compile / software | ["rounded_only", "rounded_and_residual"] | Choose one low-precision checkpoint or a paired checkpoint with a separately rounded residual. |

## Coupled constraints and scope

Finite arithmetic checkpoint pairs with explicit formats and rounding policy. The source binds A=FP32 and T=BF16 for sparse MLA training output and its backward row statistic. Other formats require their own numerical binding. This does not promise exact FP32 recovery, a universal relative error bound, less storage than one FP32 checkpoint, or an interchangeable integer codec.

Scope predicate: `p.finite_accumulator_values and p.rounded_values_and_residuals_finite and p.low_format_exactly_embeds_in_accumulator and p.cast_rounding_and_subnormal_policy_bound and p.checkpoint_coordinates_and_generation_match and p.checkpoint_visible_before_consumer and p.checkpoint_storage_survives and p.consumer_reconstructs_in_accumulator and p.reconstruction_and_downstream_error_admitted`.

- Precondition `pre_formats` (implementation): `p.finite_accumulator_values and p.rounded_values_and_residuals_finite and p.low_format_exactly_embeds_in_accumulator and p.cast_rounding_and_subnormal_policy_bound` — Bind accumulator/low formats and every cast, subtraction, addition and underflow policy; all intermediate values are finite.
- Precondition `pre_pair` (semantic): `p.checkpoint_coordinates_and_generation_match and p.checkpoint_visible_before_consumer and p.checkpoint_storage_survives` — High and residual values come from the same producer value, coordinate and generation, and survive until their consumer.
- Precondition `pre_consumer` (implementation): `p.consumer_reconstructs_in_accumulator and p.reconstruction_and_downstream_error_admitted` — The consumer adds in the accumulator format before its operation; its numerical policy admits reconstruction and downstream error.

## Local example and misuse

For a local instance of `store`, identify the object `paired_arithmetic_rounding_checkpoint` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flash-attention/delta/rounding_residual_checkpoint@1/template.yaml`. Original outline: `knowledge/studies/flash-attention/delta/rounding_residual_checkpoint@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One or two low-format arrays; the paired BF16 case uses the same bytes as one FP32 array.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: checkpoint
  level: dram
  scope: kernel
  bytes: p.elements * p.low_element_bytes
  when: encoding == 'rounded_only'
- id: checkpoint_pair
  level: dram
  scope: kernel
  bytes: 2 * p.elements * p.low_element_bytes
  when: encoding == 'rounded_and_residual'
overlap: []
```

## Obligations and numerical checks

- `arithmetic_error` (test): The actual casts and arithmetic satisfy the declared reconstruction and consumer error budget, including subnormals and range boundaries.
- `paired_visibility` (test): Every consumer reads matching high/residual coordinates and generations after publication and before reuse.
- `downstream_order` (test): Reconstruction happens in the accumulator format before downstream multiplication/reduction, with the admitted association.

Rounding the residual and the reconstructed sum can still lose information; output rounding error enters downstream row reductions and can dominate a centered gradient near cancellation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
