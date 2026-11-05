# tile_value_center_restoration@1: tile centered value with weighted mean restoration

## When to consider

When value tiles contain a large common offset, quantize residuals around a tile center and restore the center times its probability mass. Group16 mean products into matrix instructions only while all pending masses track the online row frame and the final group is flushed.

## Computation before and after

Move tilewise value centers out of narrow PV and recover them with weighted tile masses.

Choose one center mu[t,d] for each128-key tile and residual R[j,d]=V[j,d]-mu[t(j),d]. In per-channel value code units use Rhat=decode(narrow(R/s[d])) and muhat=cast(mu/s[d]). For each query, form the declared cast weights w[j] in its current online frame. Accumulate N_res[d]+=sum_j w[j]*Rhat[j,d], denominator L+=sum_j w[j], and per-tile mass U[t]=sum_{j in t} w[j]. Restore N_mean[d]+=U[t]*muhat[t,d] exactly once. With grouped16 keep up to16 masses, rescale every pending mass whenever the row reference changes, then contract the group with its matching centers into the current numerator. After insertion only the combined numerator is rescaled; retired group slots reset before reuse. Flush the final partial group before output. Native BF16 hi/lo splitting of each U and BF16 center MMA implements an admitted approximation. Final output is s[d]*(N_res[d]+N_mean[d])/L, followed by the declared output cast.

Recover the original computation: For fixed common weights, sum_j w[j]*(R[j,d]+mu[t(j),d]) equals the original weighted value sum. With quantized residuals/centers and native mass operands it becomes the explicitly admitted reconstructed-value operator.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| restore_schedule | compile / software | ["per_tile", "grouped16"] | Add each tile mean promptly or aggregate16 tile masses/means for matrix instructions. Source uses grouped16; per_tile is an algebraic alternative with its own backend obligation. |

## Coupled constraints and scope

Source VC-Attention dense fixed-length D128 MHA on Blackwell task-scheduled context, E4M3 residual V, BF16 tile means, grouped16 restoration and per-channel FP32 V scale. Per-tile restoration is an inferred alternate schedule. Causal/sparse masks, LSE and sinks are not supplied by this source route. Any key permutation must also permute V/support. This is separate from key centering, which produces a row-constant score shift, and query residual correction, which repairs scores before softmax.

Scope predicate: `(p.key_tile == 128 and p.value_features >= 1) and (p.values_decompose_as_residual_plus_one_center_per_key_tile and p.key_value_order_and_masks_match) and (p.residual_pv_and_center_mass_use_same_cast_probability_weights and p.row_denominator_uses_declared_cast_weights) and (p.old_numerator_mass_and_pending_tile_masses_rescaled_once_to_same_reference and p.pending_means_added_once_before_finalization) and (p.center_and_residual_share_per_channel_value_decode_scale and p.final_value_scale_applied_once) and (p.center_cast_residual_quantization_probability_cast_and_mass_operand_rounding_admitted and p.finite_state_and_positive_final_mass) and (p.pending_mass_center_coordinate_generation_matches_and_visible_until_mma_retires and p.selected_restore_schedule_implemented_and_resources_fit)`.

- Precondition `pre_center` (semantic): `p.values_decompose_as_residual_plus_one_center_per_key_tile and p.key_value_order_and_masks_match` — Residual V and the center have identical logical tile/head/feature identities after any common permutation.
- Precondition `pre_probability` (semantic): `p.residual_pv_and_center_mass_use_same_cast_probability_weights and p.row_denominator_uses_declared_cast_weights` — A tile center multiplies the same rounded P used by its residual PV; final mass is the declared rounded-P mass.
- Precondition `pre_frame` (semantic): `p.old_numerator_mass_and_pending_tile_masses_rescaled_once_to_same_reference and p.pending_means_added_once_before_finalization` — Pending tile masses follow every reference change until their center products are added; no contribution is lost or corrected twice.
- Precondition `pre_scale` (semantic): `p.center_and_residual_share_per_channel_value_decode_scale and p.final_value_scale_applied_once` — Centers are expressed in residual code units; one per-channel scale decodes both terms after normalization.
- Precondition `pre_numeric` (implementation): `p.center_cast_residual_quantization_probability_cast_and_mass_operand_rounding_admitted and p.finite_state_and_positive_final_mass` — BF16 centers, narrow residuals, rounded-P sums and BF16 hi/lo mass operands satisfy the selected numerical policy.
- Precondition `pre_lifetime` (implementation): `p.pending_mass_center_coordinate_generation_matches_and_visible_until_mma_retires and p.selected_restore_schedule_implemented_and_resources_fit` — Mean and mass operand groups publish together in the intended frame and remain live through the deferred MMA/tail flush.
- Constraint `c_tile` (implementation): `p.key_tile == 128 and p.value_features >= 1` — Pinned value center belongs to a complete physical128-key tile, with masks for logical tails.

## Local example and misuse

For a local instance of `contract`, identify the object `tile_centered_value_with_weighted_mean_restoration` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/tile_value_center_restoration@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/tile_value_center_restoration@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Narrow value residuals plus center operands and up to16 pending row masses.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `center_identity` (proof): Each center multiplies the mass of exactly its own logical key tile.
- `pending_frame` (test): Every pending mass follows all reference updates; insertion into N and the final group flush happen once.
- `cast_and_scale` (test): Rounded weights, BF16 hi/lo mass, centers, per-channel scale and residual rounding obey the declared recipe.
- `group_protocol` (test): Mean/mass operands have matching publication generations and survive every MMA reader.

Using unrounded probability mass for the center while residual PV uses cast P changes reconstructed values. Deferred masses in an old reference corrupt the result even when the final row mass is correct.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
