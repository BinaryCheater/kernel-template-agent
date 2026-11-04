# Adaptive sampling from a certified support

When a categorical target support is expensive to sort, sample from a superset
and remove only certified rejected weight levels. The new
[adaptive_support_rejection@1](delta/adaptive_support_rejection@1/template.yaml)
and [outline](delta/adaptive_support_rejection@1/proof.yaml) preserve the target's
original relative weights through those changing proposal sets. Paths below are
relative to the retained [tree](inputs/sources.json).

Representative route: `flashinfer/sampling.py:1365` or`:1234` ->
`get_sampling_module` custom op -> generated sampling registration ->
`include/flashinfer/sampling.cuh:860-994` or`:995-1120`. These are NVIDIA CUDA
block-scan/reduce kernels; compiled vector/block/scan menus and deterministic
options are native realizations, not different target distributions.

| Source decision and concrete binding | Rule field or boundary | Evidence |
|---|---|---|
| `TopKSamplingFromProbKernel`: sample above low; accept if full-row count strictly above sampled probability<k | `criterion=above_count` | Membership includes an entire equal-probability cohort at the k boundary. This is not exactly-k ID selection |
| `TopPSamplingFromProbKernel`: accept if original full-row mass strictly above sampled probability<top_p | `criterion=above_mass` | Original normalized probabilities define membership; after a rejection, do not substitute probabilities renormalized over survivors |
| Both kernels retain low/high/q, evaluate sampled pivot and midpoint with high, and update q to surviving original mass | `refinement=paired_midpoint`, `pre_certificates`, `pre_draws` | A rejected midpoint may remove additional classes only because its full-row certificate excludes every target token. Sampled-only cutoff is an inferred legal simpler schedule |
| Each rejection samples again using `curand_uniform(&state)*q` | Fresh conditional proposal and RNG-domain premise | The previous uniform is not automatically uniform on the newly retained support; per-round random identity must remain explicit |
| `sampling.cuh:581-654`: masked masses, block aggregate, strict inclusive-CDF>u, adjacent-difference election and atomicMin winner | Native proposal realization under `pre_numeric` | First crossing is in input-ID order; scan/reduction arithmetic and deterministic option remain bound. This is not a top-k order or sorted-probability CDF |
| Same helper records last eligible index; caller reports valid=false only when none exists, otherwise uses last-valid fallback | Explicit finite-CDF endpoint policy | Native curand uniform includes1 and finite row sums may not match1; fallback is a numerical policy, not an exact-real categorical proof |
| `:867-874,1002-1011`: seed/offset selected by output bx, Philox subsequence bx; data row may be indices[bx]; k/top_p tensor indexed by data row | Preserve output draw identity separately from row/threshold identity | Repeated data rows can represent distinct requested draws. Retaining seed/offset alone while changing output mapping does not preserve the same stream |

The invariant is that every surviving proposal set S contains the same target A
and retains original weights. Conditional on any history, an immediate accepted
proposal has weights w_i/sum_S(w). Each rejected branch preserves A; induction
on the number of rejected levels gives final law w_i/sum_A(w). Each rejection
removes at least its sampled positive-weight level, so no more than the number
of distinct rejected levels plus1 draws are needed in the exact model. A fixed
round cap with an uncertified last proposal would break this argument.

The [exact-rational check](checks/check_adaptive_support_rejection.json) covers5172
laws: widths1..4, normalized integer weights0..3 including zeros/ties, every k,
four top-p thresholds, and both refinement schedules. It computes the complete
probability tree rather than estimating frequencies. At weights[0.4,0.2,0.2,0.2],
k2 and top_p0.5 both retain all four tokens under these strict-greater cohort
predicates; an exactly-two stable-ID selector is a different support.

Two counterexamples constrain composition. At weights[0.6,0.3,0.1], k2, rejecting
the0.1 proposal leaves mass0.9. Reusing q1 with a last-ID fallback yields final
law[33/50,17/50] instead of[2/3,1/3]. For top_p13/20, the original mass above0.3
is3/5 and accepts it; renormalizing survivors changes that mass to2/3 and rejects
it. Thus both remaining proposal mass and original membership certificates are
necessary.

Existing `count_certified_topk_filter@1` certifies the completeness of a bounded
candidate list for exact selection; it does not preserve this stochastic law.
`counter_indexed_mask_replay@1` supplies invocation/counter identity discipline,
not changing-support proposals. Native CUDA/CURAND/CDF accuracy, finite endpoint
bias and mass-boundary certification remain realization obligations. Exact
proposals give the exact law; with an admitted per-proposal total-variation error
epsilon and unchanged exact certificates, at most width*epsilon follows by a
finite coupling bound.

Related designs are kept separate in the [sampling inventory](sampling-inventory.md):
Gumbel-max logits, min-p's fixed predicate, top-k-first composition, weighted
radix top-p renormalization, speculative chain correction, exact top-k and
logit-processing order.
