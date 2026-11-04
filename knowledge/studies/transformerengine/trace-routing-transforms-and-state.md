# Routing, local transforms and optimizer state

Paths below are under the retained source's `transformer_engine/common/`.
Most variants combine existing reduction, index-map and fusion mechanisms;
the interesting constraints are which values drive selection, which identities
survive packing, and which state must remain exact across updates.

| Representative and binding | Reusable choice or gap | Condition/difference |
|---|---|---|
| `fused_router/fused_topk_with_score_function.cu:138–283`: one warp per token | Fused score map → selection → probability normalization; `operator_fusion@1` and top-k/reduction rules | Softmax over all experts before top-k differs from softmax over selected logits. Sigmoid bias changes **selection**, then probabilities recover original sigmoid scores rather than subtracting bias from a rounded sum. Grouped routing scores each group by its local selected-score sum, chooses groups, then experts within their union. This is a requested routing policy, not exact unrestricted top-k. |
| `fused_router/utils.h:226–296` and selection helper | Proposed [radix_threshold_selection@2](delta/radix_threshold_selection@2/template.yaml) extends digit width and tie policy | Four-bit FP32 digits use sixteen bins and eight passes. Packed eight-bit **local** counters require at most 8160 experts; bins are extracted before wider reduction. Source picks ascending expert identities at the threshold. Signed-zero/value tie classes require a matching domain/key binding. |
| `fused_router/async_loader.h:24–70`, optimized forward `:327–355` | `persistent_tile_scheduler@1`, one/two raw input buffers, score-function specialization | Grid is capped by occupancy API's resident block estimate. Double buffering is chosen using shared-memory pressure and a four-block-per-SM heuristic. This is a performance menu; it does not change the score/selection domain. One warp owns a token and its private shared slice. |
| Forward `:45–85,207–254,1098–1129` | Top-(k+1) cutoff plus optional histogram epilogue | Drop the worst extra candidate, breaking equal-score cutoff ties by expert identity. Accumulate expert bins of cutoff minus raw score either in a later kernel or with fused atomics. Histogram reset, valid bounds and counter capacity are producer obligations. This is auxiliary calibration data, not the selected routing probability. |
| Forward bitmap output `:288–313` | `bitstream_packing@1`, one bit per expert; optional dense selected indices | Clearing prior words and preserving expert-coordinate order matter. Bitmap packing does not change probabilities or justify duplicate routes. Backward dense-map and selected-index paths use the corresponding saved selection/score contract. |
| `permutation/permutation.cu:13–126`, `triton/permutation.py:93–206` | Segmented prefix packing, inverse row map and output-owned route combine | CUDA consumes sorted route identities; Triton computes block-local route prefixes, block/expert totals and final destinations. Capacity drops set -1. Unpermute gathers all valid route contributions into the original token owner, optionally weighted; no atomic scatter is required for that ownership. FP8 no-probability source divides by topK, so it is not the plain sum path. |
| `fused_router/fused_moe_aux_loss.cu`, `fused_score_for_moe_aux_loss.cu` | Score transformation plus reduction of expert probabilities/token counts and gradient maps | Reuse fusion, partition and final reduction. Auxiliary objective coefficients and token-count domains belong to the workload; graph-safe device constants cannot be replaced by a host read. This does not establish a new top-k algorithm. |
| `fused_rope/fused_rope.cu:38–88` | Pair-coordinate rotation with shared sin/cos reused across heads | Split-half and adjacent-pair interleaving have different partner/sign maps. Only the rotary prefix is transformed; suffix copies unchanged. THD/context-parallel position origins and backward transpose rotation must use matching sequence coordinates. Generic scalar pointwise fusion alone does not encode the partner read. |
| `fused_rope/fused_mla_kv_rope_mxfp8.cu:7–53` | Load/rotate/stage once, then rowwise and columnwise MXFP8 encoding | Fixed MLA content/rotary/value dimensions 128/64/128; CTA covers 32 tokens and two heads, with six rowwise and three columnwise warps. Preserve the rotation's intermediate dtype and each scale domain. Native FP32-versus-input-type quantizer inputs are not interchangeable. |
| `activation/scaled_swiglu.cu:17–69`, `scaled_activation.cu`, grouped relatives | Gated activation/product, scale, optional clamp and quantization/dbias fusion | GLU interleave controls where paired activation/value operands live. Gradient of an activation scale needs its own reduction. Clamp limits and gate parameters are operator choices; they cannot be moved across nonlinearities merely because scale fusion exists. |
| `dropout/dropout.cu:59–100,124–167,242–248` | Logical-chunk keyed RNG plus one-bit mask and vector map | Each logical 16-value chunk initializes Philox from seed, chunk index and offset, so launch retiling that preserves chunk IDs preserves that stream. Four random words feed sixteen byte comparisons. Drop probability uses floor(256*p)/256, while multiplier remains 1/(1-p); do not infer exact Bernoulli-p expectation for arbitrary p. Input length is divisible by 16. Backward reuses the saved mask. |
| `hadamard_transform/hadamard_transform_utils.cuh:21,127–168`, `hadamard_transform.cu:26–64,325–334,487–538` | Normalized H16 plus explicit sign-diagonal choices, implemented by matrix fragments/MMA | Reuse normalized-Hadamard retrieval and contraction/layout choices, but do not call the MMA order identical to `butterfly_linear_transform@1`'s staged butterfly arithmetic. Grouped/graph-safe relatives change descriptor/work decoding; cast-fusion variants choose before/after-transform amax and quantization domains explicitly. Caller requests the transform and sign masks. |
| `multi_tensor/multi_tensor_apply.cuh:14–118` | `runtime_work_descriptors@1` tensor/chunk table plus bounded launch batches | Every CTA maps to one tensor/chunk, including a large tensor resumed after a 320-block metadata batch. Tensor-list depth changes pointer-table capacity. Zero-sized tensors must be filtered; all state lists and optional FP8 metadata share tensor identity. |
| `multi_tensor/adam.cu:113–175`, `sgd.cu:71–114`, `scale.cu`, `l2norm.cu` | Fused stateful pointwise update, chunk partial norms and later combine | Adam L2 injects decay into the gradient **before** moment updates; decoupled decay is added afterward. Master/model/moment types, bias correction and epsilon placement are prescribed. SGD first-step momentum, dampening/Nesterov and decay placement are distinct recipes. Noop behavior is functor-specific. FP8 writes consume an existing scale while collecting new parameter amax. |
| `multi_tensor/adam.cu:179–245,271–288` | New [carry_compensated_parameter_split@1](delta/carry_compensated_parameter_split@1/template.yaml) | Store BF16 model high word plus signed low-word remainder, reconstruct exact FP32 master bits before update. This uses a carry codec, not arithmetic residual addition. Its midpoint increment differs from nearest-even BF16. |
| `newton_schulz/newton_schulz.cpp:257–327` and execution tail | External cuSolverMp matrix iteration | Source binds column sharding, m≤n, per-iteration coefficient triples, normalization and workspace ownership. Caller/internal streams synchronize through events; registered workspace is reused by configuration. Device polynomial/GEMM schedule is external, so no in-repository Newton–Schulz kernel Template is claimed. |

## Preserved identities and deliberate recipe choices

With routes token 0→experts A/B and token 1→B, packing creates three rows.
An output-owned unpermute reads token 0's two destinations and token 1's one;
dropping one destination sets its inverse entry to -1. It is a weighted sum only
when that is the requested output rule; the FP8 averaging branch is separate.

The parameter split stores raw low bits. If the low sign bit is one, the model
high word is incremented and reconstruction decrements it again. The
[bit-codec check](checks/check_parameter_split.py), run with
`.venv/bin/python knowledge/studies/transformerengine/checks/check_parameter_split.py`,
exhausts every low word for eight selected finite-domain high words and records
[524288 successful roundtrips](checks/check_parameter_split.json). It also
distinguishes the even-high midpoint from nearest-even conversion. This resolves
the format identity/tie question; native alias layout, pair publication and a
complete optimizer run remain separate obligations.

Version 2 of radix threshold selection preserves version 1's key width, rank
invariant, exact quota and numerical premises. Migration is `digit_bits=8`,
`tie_policy=any`, with the old arbitrary-tie contract supplying the new
tie-emission premise. New four-bit bindings choose their counter width/capacity;
smallest-index bindings additionally establish matching key/value tie classes
and ordered threshold ties. Version 1 and all its source evidence remain intact.
The rank proof is unchanged for radix 16 versus 256; packed counter extraction
and native deterministic tie emission remain realization obligations.

## Additional row loss and small mixing operators

`triton/cross_entropy.py:47–124,128–164,205–271` streams max/exp-sum state and,
for label smoothing, the raw logit sum. It records the target logit separately.
Tensor-parallel pre/post kernels merge per-shard mass in a common max frame and
recover the target only from its owning vocabulary shard. Ignore-label counts,
global vocabulary size and optional argmax identity/NaN policy are observable.
Existing online reduction and split merge supply the algebra; target ownership
and saved/recomputed logits bind backward. No full probability matrix is needed.

`triton/mhc.py:806–864,1016–1080` implements n=4 Sinkhorn-like mixing in the log
domain: alternate row and column logsumexp potentials, then exp(f+x+g). Forward
can retain all potentials for backward or recompute the iterations. This differs
from TileLang's epsilon-regularized value-domain sequence, so record algorithm,
iteration and dtype as the workload binding. Projection/square-sum fusion,
aggregation and expand-combine reuse contraction/reduction choices. A fixed
number of iterations is not an exact-convergence claim. `triton/pad.py` adds
guarded data padding, already covered by storage/tail rules.
