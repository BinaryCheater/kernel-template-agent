# FMHA nonidentity pruning — error certificate and control path

When small attention-score tiles are candidates for omission, a maximum-weight
threshold should propose work removal; it cannot certify normalized output error
by itself. Retrieve
[bounded_normalized_weight_pruning@1](delta/bounded_normalized_weight_pruning@1/template.yaml)
with its [outline](delta/bounded_normalized_weight_pruning@1/proof.yaml) when a
contract admits absolute error and conservative retained/dropped mass plus value
range bounds are available. Keep the certificate independent of native skip-flag
publication and pipeline retirement.

This continues [FMHA decisions](trace-fmha-contracts.md) and
[reference/gain frames](trace-fmha-frames.md). Source ranges refer to forward
`fmha.py` in the [retained tree](inputs/sources.json). Full FMHA forward masks,
TMEM layouts/lifetimes, scheduler/resource paths and the backward entry remain
active. The remaining attention groups are still assigned.

## Certificate and composition

For one visible output row, retained nonnegative weights sum to K>0 and dropped
weights to D≥0. With weighted means y_keep and y_drop,

`y_full − y_keep = D/(K+D) * (y_drop − y_keep)`.

If every coordinate lies in an interval of diameter R, the absolute infinity-norm
output error is at most R*D/(K+D), multiplied by the absolute output scale. For
K_lower≤K and B≥D, replace the ratio by B/(K_lower+B). Add independently admitted
arithmetic/conversion error. Natural LSE error is log1p(D/K), bounded by
log1p(B/K_lower) plus its additional numerical error. Base2 LSE needs an explicit
unit conversion. D0 has zero pruning error without needing a dropped mean.

For an omitted tile with n valid weights and maximum natural logit u at reference
r, `n*exp(u-r)` bounds its mass. A ratio threshold tau also gives `n*tau` when the
source comparison passes. **Accumulate every omitted tile**, and rescale B and
K_lower together whenever the online reference or common gain changes. Sink terms
are zero-valued denominator contributions and must be included in the value
interval and common frame. A small per-element bound without n, accumulated mass
and V range is insufficient.

One partial composition is: certify an approximate restriction of the original
weighted operator using this new rule, then apply
`online_rescaled_reduction@2` exactly to the resulting kept operator under its
own precision/frame premises. The second rule's no-dropped-terms premise now
refers to that kept operator; it does not certify the original omission. Source
threshold-only execution has no such accumulated certificate and remains a
conditional instance rather than an admitted implementation of the new rule.

## Source decisions and boundaries

| Source decision | Recovered binding | Rule field or missing obligation |
|---|---|---|
| `calculate_skip_softmax_flag`:1957–1977 | A live row proposes skip if `(tile_max-old_reference)*scale_log2 < log2(tau)`. Out-of-range query rows vote skip. Every warp votes all lanes and all four softmax warps must agree. | `ratio_threshold` proposal plus `collective_skip_uniform`. Every visible row in a collective must also satisfy its certificate; no output contract is attached to padded rows. Native vote packing/visibility remains required. |
| Host setup:3777–3779 | None/nonpositive tau disables skipping; any positive tau is converted to log2. | This candidate binds0<tau≤1. Larger source inputs are outside its declared proposal domain. No host numerical certificate is generated. |
| `softmax_step`:2514–2580 | Normal path updates P/sum, publishes skip flag0 and commits P; skipped path preserves the old row sum, consumes the in-place dependency, publishes flag1, fences TMEM and commits its P pipeline generation. | Arithmetic bypass does not bypass ownership. `skipped_work_protocol_retired` still requires native handle incidence, generation and lifetime validation. Presence of calls alone is not a progress proof. |
| `mma_pv`:1874–1912 | Wait P, read the shared decision, issue no PV MMA on skip, then release P and commit O regardless. | A committed O generation may preserve prior state rather than contain new arithmetic. `initial_kept_state_valid` prevents interpreting an uninitialized accumulator as a valid retained result. Full first/masked/tail schedule remains open. |
| Host reference:3905–3956 | Block maxima compared against the stored reference; when a collective is kept, reference updates to the maximum unconditionally. | Does not apply the device's optional bounded-lag reference policy. This mismatch is separate from error certification. Neither reference path proves closeness to the complete unpruned operator. |
| Missing source certificate | No K_lower, accumulated dropped-mass B, V diameter or absolute output/LSE bound is computed by these paths. | `kept_mass_lower_valid`, `dropped_mass_upper_valid`, `coordinate_value_ranges_valid` and numerical-budget constraints must be supplied by a later certificate/backend. Source tau is not substituted for them. |

The old `masked_tile_pruning@1.pre_neutral` permits removing only identity
contributions. This new rule restricts nonzero terms under an explicit approximate
contract, so it has a separate identity. `approximate_math_rewrite@1` changes map
arithmetic and supplies no restriction/certificate choice. The prior online frame
revision remains unchanged.

## Discriminating checks

[Check](checks/fmha_pruning.py) verifies10449 exact rational weighted-sum domains,
including zero individual weights, signed values and every nonempty kept subset
with positive mass.2484 cases attain the output bound.31347 common-frame checks
show retained/dropped bounds cancel identical positive frame factors. Looser mass
bounds stay conservative; undercounting a tight dropped mass violates the claimed
bound.16 scope-premise mutations reject, as do output budget1 and natural-LSE
budget0.01 for the source-shaped K1/B0.125/R1000 witness. Output budget112 and
natural-LSE budget0.125 admit that separately supplied certificate; this does not
claim the source computes or conservatively rounds it.

The check also executes retained device skip/freeze expressions and the scalar
comparison inside the host block reference. For identical live rows, base2 block
maxima(0,4,−4), lag8 and log2(tau)=−5:

| Block | Device reference before | Host reference before | Device skips | Host skips |
|---|---:|---:|---|---|
| 0 | −inf | −inf | No | No |
| 4 | 0 | 0 | No | No |
| −4 | 0 | 4 | No | Yes |

The device retains reference0 after the second block while the host updates to4.
This is an executable source-expression discrepancy on a uniform-row case, not
an observed GPU failure. It shows why a host reference must use the same reference
policy if it is intended to check the source pruning decisions.

AST control-path checks locate P flag publication/fence/commit in the skip branch,
PV MMA under the not-skipped guard, and unconditional P release/O commit. They do
not establish the full in-place handle protocol or native progress. The
[result](checks/fmha-pruning-result.json) retains counts, a tight undercount witness
and the host/device decision sequence. General quantization/approximation error,
outward-rounded certificates and unchanged gradients are not proved here.

Next: full mask and empty-row behavior, in-place score/probability handshakes,
TMEM storage and forward scheduling, followed by backward and the remaining
Blackwell attention groups. No FMHA entry is marked source-complete yet.

Continued in the [mask/channel trace](trace-fmha-masks.md): host empty-row
filtering is incomplete and the in-place notifications use asymmetric boundary
counts. These do not discharge the pruning rule's positive mass/native retirement
premises.
