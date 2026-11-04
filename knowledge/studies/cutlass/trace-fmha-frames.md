# FMHA reference/gain frame and compensation

When online softmax's maximum changes trigger accumulator correction, consider a
finite reference that can lag the maximum. Preserve the normalized-weight frame,
not the stronger assertion that the reference is always the exact maximum. If P
is multiplied by a common gain to use more low-float range, **every** numerator
and denominator contribution must share it, including attention sinks; LSE must
remove its logarithm. [online_rescaled_reduction@2](delta/online_rescaled_reduction@2/template.yaml)
and its [outline](delta/online_rescaled_reduction@2/proof.yaml) make those choices
and rejecting premises explicit.

This continues the [forward decision slice](trace-fmha-contracts.md). All source
ranges refer to `fmha.py` in the retained FMHA directory. The FMHA pair, remaining
attention groups and whole CUTLASS inventory are still active. No native math,
TMEM execution, progress, mask or backward assurance is claimed by this algebraic
candidate.

## Recovered decisions and invariant

Work in scaled natural-logit units x, let finite reference r represent raw source
row reference times the positive softmax scale, and let G=2^gain_log2. The state
invariant is L=G*sum(exp(x−r)), O=G*sum(exp(x−r)*v). With a new reference r',
correct only the old contributions by exp(r−r'), then add the next tile in the
new frame. The final O/L is independent of r and G; natural LSE is
r+log(L)−log(G). A lag changes the representation, while dropping a nonzero tile
changes the operator and is excluded by this rule.

| Source decision | Actual binding | Rule field or obligation |
|---|---|---|
| Setup:349–357 | Correction disabled: lag0, FP8 gain exponent8 (gain256). Correction enabled: lag8, exponent0 (gain1). Combined maximum exponent8 fits E4M3 finite448 in the intended positive-scale bounded-lag frame. | `reference_policy`, `threshold_log2`, `weight_gain_log2`; range is still an explicit guard rather than a blanket FP8 theorem. Non-FP8 ignores P gain in source. |
| `softmax_step`:2454–2466 | Preserve old reference for scaled base2 increases≤threshold; store old/new reference pair. | `lagged_max` policy. This changes version1's exact-max invariant; the old body is preserved and a versioned extension is used. |
| Probability transform:2473–2476, sum:2507–2508,2516–2517,2554–2558 | Add gain exponent to the probability exponent for FP8. Scale previous sum by half the reference correction, duplicate into two packed accumulators, add the two results. | Both old-sum halves combine to exactly one correction in real arithmetic. The factor0.5 is not a separate half-weight semantic. Native packed operations and P cast error remain backend obligations. |
| Correction:3043–3049 | If old/new reference differs, multiply old O by the same exp2 correction; unchanged reference avoids this pass. | `implementation_order` and matching numerator/denominator references. Full TMEM scheduling still must establish which O generation contains the old contributions. |
| Finalizer:3220–3231,3256–3267 | Normalize O using row_sum. Compute base2 LSE from log(row_sum)+scaled reference, then subtract FP8 gain exponent. | `finalize_log_compensation_exact`; ordinary weights' common gain cancels, and LSE removes it explicitly. |
| Sink:3221–3230; host sink:3672–3683 | Sink is an already scaled logit, added only to the denominator as exp(sink−reference). Source does **not** multiply it by G. Host passes its ordinary FP32 value. | `extra_denominator_terms_share_frame` fails for source sink plus nonunit P gain. No current source guard excludes that combination. It is not admitted by this candidate binding. |
| Optional tile skip:2514–2580,1881–1897 | Skips denominator and PV terms while still advancing protocol handles. | Fails `no_nonidentity_terms_dropped`. The threshold/error rule remains a separate pending mechanism, not hidden inside frame selection. |

## Discriminating source and algebra checks

[Check](checks/fmha_frames.py) executes retained setup and finalizer expressions,
checks192 complete recurrence domains/384 prefix invariants and12 packed-sum
cases, validates the new schema/outline/guards and preserves all five old variable
domains. Both traversals, reference lag0/8, gain1/256, optional sink and signed
weighted values are included. [Result](checks/fmha-frames-result.json) retains
these domains. [Migration](delta/online_rescaled_reduction@2/migration.md) maps
old fields/obligations and explicitly records the positive-mass domain restriction
and reopened cost/backend proofs. Existing version1 is unchanged.

A minimal sink witness has one score0/value1 and sink logit0, output scale1.
With gain1 the source expressions yield output1/2 and base2 LSE1, as requested.
With FP8 gain256 the source ordinary sums are256, but sink adds only1: source
expressions produce output256/257≈0.996109 and LSE log2(257)−8≈0.005625, versus
requested1/2 and1. These are source-expression/scalar results, not an observed GPU
failure. Probability256 and value1 are exactly representable in E4M3; the missing
common-frame factor is distinct from a generic rounding issue. No code outside
the study is changed.

All14 explicit scope-premise mutations reject. In particular, disabling common
sink-frame compensation, permitting nonidentity pruning or removing positive
finite final mass rejects the binding. Lag/range and numerical admission remain
required even in the source branch that avoids the sink mismatch.

Version2 exposes reference policy and common gain together because the same
framed-state invariant owns their correction, extra-term and LSE obligations.
The largest-reference-update count and source warp ballot incidence are not
modelled as a universal cost: the proposal leaves those effects open rather than
copying version1's unconditional estimate. This is not a promised speedup.

Next: bounded value-based pruning, full mask/empty-row policy and the TMEM/skip
flag protocol, then forward resource/layout scheduling and the backward entry.

Continued in the [pruning trace](trace-fmha-pruning.md): the new restriction rule
certifies complete dropped mass and value ranges, while the existing source has
no such certificate and a frozen-reference/host-mask discrepancy is retained.
