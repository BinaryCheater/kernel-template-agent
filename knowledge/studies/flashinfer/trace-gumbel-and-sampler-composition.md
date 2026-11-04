# Logits noise and sampler composition contracts

A sampler's output law, chosen support and exact random-stream mapping are
separate contracts. Switching between a compact support and a full-row scan can
preserve an ideal distribution without preserving same-seed outputs; changing
tie support may change even that distribution. Paths below refer to the retained
[tree](inputs/sources.json).

| Source choice | Concrete reuse or boundary | Evidence |
|---|---|---|
| `include/flashinfer/sampling.cuh:679-735`: generate4 Philox uniforms per counter group, scale by1-FP32epsilon, transform via two log2 calls, cast noise to DType | `approximate_math_rewrite@1(rewrite=fast_log,fused_fma=false)` addresses admitted native log approximation; RNG mapping remains explicit | Finite uniform scaling excludes1 and bounds noise. Casting noise to BF16/FP16 before addition is part of the recipe; it is not the same as FP32 perturbation followed by output cast |
| `:737-788`: perturb original logits, keep one candidate/thread, block-reduce by LogitsArgMaxOp | Declared lexicographic max(score,position), with equal values preferring larger position | Explicit comparator makes ties independent of reduction grouping on its admitted non-NaN domain. Scalar `max` alone would lose the returned-ID policy |
| RNG subsequence base uses output bx*d+vector start, plus local4-value group offset; offset/seed selected per output row | `counter_indexed_mask_replay@1` supplies identity discipline only after a full counter/component map is bound | DType/vector-width changes may regroup the same vocabulary coordinate into another counter/component. Row indices can map several independent output draws onto the same data row |
| `flashinfer/sampling.py:1636-1730` selects sorted exact top-k values/IDs, normalizes compact values or applies compact softmax, then top-p samples and gathers original ID | Reuse exact selection + normalization + the top-p support contract + original-ID gather | Per-row k uses caller-supplied top_k_max to size selection and then masks row tails. The supplied bound must cover requested row k; compact order and selected tie IDs matter |
| Full-row top-k rejection uses strictly-greater count<k | `adaptive_support_rejection@1` whole-cohort contract | Compact exactly-k selection is not equivalent on a boundary tie unless the requested policy resolves that difference |
| `sampling.cuh:1124-1218`: min-p computes row max, fixed pivot=min_p*max, retained mass and masked inverse-CDF with x>=pivot | Fixed predicate plus categorical proposal, not adaptive above-count/mass refinement | Inclusion uses>= at the threshold; zero weights and finite last-valid fallback need their explicit support policy |

For ideal real arithmetic and v in(0,1), the source transform
`-ln(2)*log2(-log2(v))` equals the usual `-ln(-ln(v))` plus the common constant
ln(ln2). The common shift does not alter a real-valued argmax. The source also
changes v through finite uniform scaling and rounds noise to DType; those steps
are independent numerical policies, so the real identity does not establish
native exact categorical equivalence. All-negative-infinity logits and padded
argmax positions have an explicit source fallback domain, not a normalized
categorical distribution.

A boundary-cohort counterexample is already checked in the
[adaptive probability evidence](checks/check_adaptive_support_rejection.json):
weights[0.4,0.2,0.2,0.2],k2,top_p1 retain all four under the full-row strict-greater
predicate. Compact exactly-k selection keeps0.4 and one0.2, producing a two-ID
law[2/3,1/3] instead of the original four-ID law. The fast-path shape dispatch has
no observed tie-cohort certificate, so its source comment about distribution
equivalence does not supply that boundary premise.

Even without ties, same-seed equality is stronger than distribution equality.
For row[0.3,0.1,0.6],k2,top_p1, the full masked CDF visits IDs0 then2 with masses
1/3 and2/3; the sorted compact CDF visits IDs2 then0 with masses2/3 and1/3.
The same uniform0.2 selects ID0 or ID2 respectively. Both have the same ideal
unconditional law when all other premises match. Source compact logits/probs
paths may align with each other through identical sorted values; this does not
imply alignment with the unsorted full-row fallback.

No extra Template is needed for the existing normalization/gather composition.
The Gumbel-max law as a selectable sampler, native finite-grid error, exact
counter/component codec and zero/NaN/negative-infinity policy remain named
contract/realization gaps. Weighted AIR top-p, exact fixed/varlen top-k and
logit processing remain in the [sampling inventory](sampling-inventory.md).
