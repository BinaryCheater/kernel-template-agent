# Deterministic bucket sums and typed logit pipelines

When positive FP32 histogram entries share an exponent, sum integer fraction bits
and counts, then reconstruct once. This trades floating atomic order dependence
for a specified final rounding policy. The new
[shared_exponent_integer_sum@1](delta/shared_exponent_integer_sum@1/template.yaml)
and [outline](delta/shared_exponent_integer_sum@1/proof.yaml) capture this choice.
Paths below refer to the retained [tree](inputs/sources.json).

| Source decision | Concrete binding / boundary | Evidence |
|---|---|---|
| `air_top_p.cuh:137-194`: extract23 fraction bits, accumulate uint64 mantissa sum M and separate count N | `aggregation=cta_then_global`, `pre_census` | Every value contributes count, including a power of two whose fraction is zero. Empty bins are not identified by M0 alone |
| Initial11-bit FP32 radix bins fix exponent; later passes retain their previous prefix | `pre_domain`, `pre_exponent` | One common exponent E is necessary. Mixed-exponent or signed inputs cannot use this decoder; negative zero requires canonicalization outside the rule |
| `:229-282`: CTA-local histogram/count arrays, then exact global integer additions | Native source uses CTA partials; `direct_global` is an inferred alternative under the same census/completion premises | Both channels must publish before reconstruction. Integer atomics need no compound pair update if readers wait for completion |
| `:157-181,326-349`: reconstruct hidden-bit carries and discard excess fraction bits, bypass N0 bins | One final nonnegative round-toward-zero, not nearest-even | Exact significand S=M+N*2**23 for E>0, or M for E0. Exact real sum is S*2**(max(E,1)-150). Empty normal bins must bypass this helper |
| `csrc/renorm.cu:43-56`: vocab<2048 uses another threshold path, otherwise deterministic/non-deterministic FP32 AIR | Registered scope; source threshold route and numerical policy remain visible | Generic header types do not establish an additional registered dtype |

The [independent rational-oracle check](checks/check_shared_exponent_sum.json)
compares the source reconstruction with a binary search for the greatest FP32
number not exceeding the exact rational sum. It covers750 small shared-exponent
cases and45 large-count cases through INT32_max, including zero mantissas and
subnormal-to-normal carry. A source-style empty normal-bin call can yield a
nonzero result, which is why the wrapper's N0 bypass is part of the contract.

For values1 and1+3*2**-23, reconstruction returns2.000000238418579 whereas
nearest-even returns2.000000476837158. Another common-exponent four-value case
has floating-order results5 and5.000000476837158, while integer reconstruction
always returns5. Deterministic bucket bits therefore do not prove unchanged
exact-real top-p membership. The separately extracted
[mass-ranked boundary](trace-weighted-radix-and-selection.md) still needs its
native mass-comparison/rounding premise.

The remaining classic softmax and logit-pipeline mechanisms have concrete reuse:

| Source design | Rule binding / boundary | Evidence |
|---|---|---|
| `sampling.cuh:302-421`: one CTA's running max and rescaled denominator, with optional scaled-input SMEM cache | `online_rescaled_reduction@2` max/frame and mass equations; local norm-map fusion and separate input residency choice | Cached path reuses the same scaled DType stage; uncached path reloads/scales. Required stage precision must match, rather than assuming arbitrary wider cached values are equivalent |
| `:423-580`: disjoint input slices publish raw(max,denominator), separate kernel merges pairs and normalizes original coordinates | Same max/frame and mass algebra; publication-dependent kernel split | These are raw unnormalized mass pairs, not attention normalized-output/LSE records. Float2SoftmaxReduceOp`:133-141` bypasses infinite-max identities; valid finite logits or masked-negative-infinity support must be explicit |
| `:336-343,573-576`: temperature multiplication uses __fmul_rn before subtraction/exp | Preserve noncontracted FP32 multiply checkpoint | An FMA that fuses temperature scaling and max subtraction changes rounding. Positive finite temperature is a public contract; a native zero-temperature fallback is not a general API extension |
| `flashinfer/sampling.py:943-1026`: public route chooses generated Blackwell softmax or classic CUDA fallback | Target/alignment/row-width/PDL menu, not a free semantic rewrite | Generated row-cached cluster bodies and their final native state maps remain named realization follow-ups |
| `logits_processor/legalization.py:29-78`, `compiler.py:25-112`: legalize processor order against LOGITS/PROBS/INDICES types, validate, then replace adjacent guarded patterns by priority | Existing fusion dependency/retained-stage obligations; typed adjacency is necessary but not a numerical proof | Compiler does not authorize arbitrary commuting/reordering of typed operators |
| `fusion_rules.py:50-130`: Temperature+Softmax, TopK+Sample, TopP+Sample, MinP+Sample; triple TopK/TopP/Sample only with joint_topk_topp opt-in | `operator_fusion@2` local scalar/norm edges plus separately requested sampling law | Joint predicates differ from top-k renormalization followed by top-p. Opt-in changes the requested support contract; do not infer equivalence merely from matching classes |
| `operators.py:41-74,356-402`: separate Temperature divides then hands off a typed tensor; fused path calls public sampling.softmax | Preserve required scale/cast checkpoints and runtime/default parameters | Same types do not erase reciprocal/division, materialized dtype or final RNG differences. Fused dispatch still follows the public target menu |
| `operators.py` / `processors.py`: Temperature, Softmax, TopK, TopP, MinP, Sample and fused wrappers | Actual retained processing menu | No penalty/bias operator is inferred from the namespace. Custom registered rules need their own semantic guard and parameter/lifetime binding |

One composition is typed temperature scaling -> softmax -> requested support
transform -> categorical sampling. Preserve each edge's dtype, normalization
frame, support/tie policy and RNG identity. A pipeline wrapper may remove
intermediate materialization only when those premises remain true; graph shape
alone is insufficient. Earlier [composition witnesses](trace-gumbel-and-sampler-composition.md)
show same-uniform ordering and boundary-tie differences.

A further distinct source contract is now identified in `flashinfer/cake_sampling.py:16-55`:
cluster radix exact-k -> sorted sparse slab -> top-p/sampling with row-relative
64-bit fixed-point prefix masses. Its contract drops entries below2**-53 times
row maximum and defines explicit invalid/infinite/zero-row policies. This is
not the same-exponent integer histogram above. Native fixed-point codec and
slab/prefix boundary semantics are the next representative in the
[sampling inventory](sampling-inventory.md); the entry comment alone is not a
verified native realization.
