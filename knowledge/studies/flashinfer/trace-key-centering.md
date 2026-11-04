# Common key centering before attention quantization

The selected route is `prepare_fp8_attention(..., backend="cake")` ->
`flashinfer/experimental/cake_fp8_attention/cake_backend.py` -> four generated
CUDA stages in the retained [revision](inputs/sources.json). The
[stage-source map](inputs/centered-attention-sources.json) provides the complete
revision-relative files. This trace calls them stats-partials, stats-finalize,
quantize and attention rather than treating generated file identities as designs.

New choice: [softmax_key_centering@1](delta/softmax_key_centering@1/template.yaml)
and its [proof](delta/softmax_key_centering@1/proof.yaml).

| Observed decision / source binding | Candidate or existing field | Evidence and boundary |
|---|---|---|
| Stats-partials accumulates K channel sums and abs-V maxima per128-row block; stats-finalize :63-90 folds all blocks in fixed order and divides K sums by S | `centering=channel_mean`, `statistics_and_generations` | One FP32 mean per(batch,head,channel), not per64-key scale block. Statistics exclude capacity padding. Source dense head128,S>=1 returns no LSE. |
| Quantize :140-164 loads common mean and subtracts it before selecting K maxima; invalid rows become zero codes | `pre_center`, `pre_scores`, `pre_generation` | Every valid key in a normalization row receives one common translation. A reusable runner quantizes changing values every launch; it cannot cache stale means/scales by shape alone. |
| Quantize :164-198 derives two K amax values and positive floors, stores amax/127 scales | `pre_numeric`; block_scaled_partial_accumulation@1 scale-constancy invariant | Q scale is per128-row block; centered K scales are per64-key half. Different local encoding scales do not change the shared center's domain. Native scale application must use the correct score columns. |
| Attention :254,335-336 reads Q scale and each64-key half's K scale | Score arithmetic binding | Restore the decoded INT8 dot with the matching Q/K scale product before softmax. No mean compensation is added to output: ideal key translation is a row-constant logit shift. |
| Attention :616-621 chooses running max, correction and bias=8-max; :952-959 sums FP32 weights before E4M3 cast | online_rescaled_reduction@2 `reference_policy=running_max`, `weight_gain_log2=8` | Both row mass and PV weights carry gain256. This is common gain, unlike numerator_weight_gain@1. Final division cancels the gain; original LSE would require both common-gain compensation and key-center offset. |
| Attention :1084-1112 drains each zero-initialized FP8 PV tile then folds into FP32 running O | Per-tile low-precision accumulation realization | Reinitializing tile PV confines that instruction's accumulation error to a tile, but does not prove its precision model. Source README truncation/emulation claims remain source claims, not device validation here. |
| Attention :1167-1209 sums row state, applies per-channel V scales and casts BF16 O | Explicit scale/rounding binding | V scale is constant over keys for that output feature and may be factored at final output. No original-frame unnormalized score/LSE consumer exists in this source API. |

For ideal affine scores x_j=a*q dot K_j+b_j and common mu, centered scores
are x_j-c with c=a*q dot mu. Softmax is unchanged; original natural LSE equals
centered LSE+c, and original base2 LSE adds c*log2(e). Bias/masks keep their
original coordinates. Nonlinear logit clipping does not generally commute with
this shift. A denominator-only sink must shift by -c as well; an untouched sink
changes relative mass even though ordinary key probabilities share a shift.
Source has no sink, so this is a rule applicability boundary rather than a claim
of extra source functionality.

The K mean need not be mathematically exact for the ideal identity: any shared
finite vector gives one row constant. It still affects actual subtraction,
cancellation, scale selection and rounding. Quantization is not translation
equivariant, and a shared mean is not a proof that error always improves.
Likewise, changing Q quantization adds its own score error; row-offset
restoration must use the same selected query operand and intended score units.

The [logical check](checks/check_key_centering.json) covers three query/support
rows, masked subsets, restored LSE, a wrong independently centered-key-block
path and a sink shifted inconsistently. INT8 nearest-even scalar coding of
BF16-representable keys[1024,1032,1040,1048] has maximum reconstruction error
about0.756 without centering and0.0315 after shared centering. This is one range
witness, not a full128-feature quantizer or a speed/accuracy measurement. Native
mean/subtraction, half-block scales, code maps, MMA and exp/reciprocal arithmetic
remain explicit obligations.

online_rescaled_reduction@2 changes the exponent reference **after** scores
exist; it does not choose the operand translation before quantization.
nonnegative_scale_hoisting@1 moves a multiplicative ReLU scale; it cannot supply
an additive key shift. The new rule keeps these choices separate and states
which non-normalized consumers would need compensation.

This dense quantized recipe now has computation, scale, pipeline and numeric
boundary dispositions. Continue the sparse quantized/model routes and remaining
target variants; do not expand its SM90 dense/no-backward API by analogy.
