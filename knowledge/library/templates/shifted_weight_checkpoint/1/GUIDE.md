# shifted_weight_checkpoint@1: reference shifted exponential tile checkpoint

## When to consider

When a later pass needs probabilities and the forward already forms exponent tiles before final normalization, save those tiles with the exact references used to form them. Decode each as W*exp2(reference-final_LSE2). Tiles may use different running references. This trades full narrow weight storage plus references for score recomputation; normalizing with the wrong tile reference or treating saved W as already normalized changes the result.

## Computation before and after

Store pre-normalization exponential tiles with their own references, then decode with the final row normalizer.

Use base2 scaled logits z. For tile t of row i, save W_ijt=cast_T(exp2(z_ij-r_it)) and its actual reference r_it, instead of waiting for final normalization or later recomputing scores. The reference may be a running or lagged reference; it need not be the tile maximum. Once final row L2_i=log2(sum_j exp2(z_ij)) is available, decode P_ij=float_A(W_ijt)*exp2(r_it-L2_i). An empty tile may use defined zero weights and a finite safe reference; otherwise tag it and decode zero without evaluating -inf minus -inf or reading undefined payloads. Convert a source unscaled reference m to r=score_scale*log2(e)*m and a natural LSE to L2=LSE*log2(e). Retain the A-format decoded P for pointwise consumers; convert separately to T for an MMA consumer when needed. A final denominator-only sink is included by L2 without changing ordinary stored weights. Do not replace each tile's reference by the final reference without correcting its stored weights, or apply normalization twice.

Recover the original computation: In real arithmetic remove the saved reference using exp2(z-r)*exp2(r-L2)=exp2(z-L2); under the numeric policy decode the same logical normalized weights with the declared checkpoint error.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Deterministic exponent-weight checkpoints with positive finite final row mass, known score/reference units and defined empty-tile behavior. Source twoCTA sparse MLA saves BF16 unnormalized P plus FP32 per-row/tile references, then decodes in FP32. No full score recomputation, arbitrary dropout replay, zero total-mass normalization or bitwise equivalence to once-rounded normalized P is provided. Extra reference storage is explicit.

Scope predicate: `p.scaled_score_units_and_log_base_known and p.each_saved_tile_has_its_actual_exponent_reference and p.checkpoint_support_matches_final_normalizer and p.final_row_normalizer_positive_finite and p.nonempty_tile_reference_finite and p.empty_tile_decode_defined_without_undefined_reference_arithmetic and p.saved_weight_and_reference_coordinates_generations_match and p.masked_checkpoint_weights_defined_zero and p.saved_state_visible_and_alive and p.exponent_cast_scale_and_decoder_range_admitted and p.decoded_weight_error_and_consumer_casts_admitted`.

- Precondition `pre_frames` (semantic): `p.scaled_score_units_and_log_base_known and p.each_saved_tile_has_its_actual_exponent_reference and p.checkpoint_support_matches_final_normalizer` — Every tile records the reference actually used to form its weights, in known units, for the same logical row/support as the final log-normalizer.
- Precondition `pre_mass` (semantic): `p.final_row_normalizer_positive_finite and p.nonempty_tile_reference_finite and p.empty_tile_decode_defined_without_undefined_reference_arithmetic` — The whole row has positive finite mass; nonempty tiles have finite references, and empty tiles have defined zero decoding through a finite safe frame or an explicit empty branch.
- Precondition `pre_values` (semantic): `p.saved_weight_and_reference_coordinates_generations_match and p.masked_checkpoint_weights_defined_zero and p.saved_state_visible_and_alive` — Weight/reference/LSE generations and coordinates match, excluded weights are defined zero, and all checkpoints survive publication until consumption.
- Precondition `pre_numeric` (implementation): `p.exponent_cast_scale_and_decoder_range_admitted and p.decoded_weight_error_and_consumer_casts_admitted` — The numerical policy covers exponent approximation, narrow weight/reference/LSE storage, decoding scale and consumer conversion, including underflow.

## Local example and misuse

For a local instance of `store`, identify the object `reference_shifted_exponential_tile_checkpoint` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flash-attention/delta/shifted_weight_checkpoint@1/template.yaml`. Original outline: `knowledge/studies/flash-attention/delta/shifted_weight_checkpoint@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: Save forward-generated exponent tiles for later consumers without recomputing scores.. Storage: Narrow full weight checkpoint plus per-row/tile references and final row LSE.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: saved_weights
  level: dram
  scope: kernel
  bytes: p.weight_elements * p.weight_element_bytes + p.row_tile_count * p.reference_element_bytes
    + p.row_count * p.normalizer_element_bytes
overlap: []
```

## Obligations and numerical checks

- `frame_identity` (proof): Each saved weight and its actual tile reference decode against the final matching row normalizer, including any denominator-only sink.
- `coordinate_and_empty` (test): Tile/head/token/reference coordinates and generations agree; masked weights and empty tiles decode to zero through safe finite frames or an explicit branch, without undefined reads or arithmetic.
- `checkpoint_numeric` (test): Saved narrow weights, reference/LSE precision, exponent range and separate consumer casts satisfy the numerical contract.

Rounding unnormalized weights before applying the final scale differs from rounding normalized probabilities once; tiny saved weights may vanish before later rescaling.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
