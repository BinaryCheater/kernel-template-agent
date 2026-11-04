# Earlier NVIDIA Triton attention designs

The retained flash_attn/flash_attn_triton.py and flash_attn_triton_og.py are
NVIDIA paths separate from the excluded AMD backend. Their source notes name
historical Triton compiler constraints. This pass reads their computation and
ownership choices without treating them as current compiled/runtime evidence.
Paths below are relative to flash_attn/ in the [retained tree](inputs/sources.json).

| Source decision | Rule / gap | Concrete binding and boundary |
|---|---|---|
| flash_attn_triton_og.py:70-115: causal forward | reduction_partition@1 and output_tiling@2; normalized-accumulator recurrence gap | Each tile updates m and l, rescales old normalized O by alpha*l_old/l_new and current P by beta/l_new, then adds PV. Thus O is normalized after every iteration, not an unnormalized numerator with one deferred divide. online_rescaled_reduction@2's current finalize menu does not directly encode this representation. |
| flash_attn_triton_og.py:120-141,219-242 | Saved max/denominator and scaled-gradient backward binding | Forward saves(m,l); preprocess divides dO by l and sets D=dot(O,dO/l). Backward uses raw P'=exp(score-m), so P'*dO/l and P'*(dO*V/l-D) recover normalized probability derivatives. Treating P' as normalized without the gradient scaling would multiply by the denominator. A conceptual LSE=m+log(l) translation is possible but not the source's actual memory/numerical path. |
| flash_attn_triton_og.py:175-246,310-345 | reduction_partition@1 assignment contiguous_sequential; owned output map | One backward program per batch/head visits key tiles serially, accumulates dK/dV per key tile and read/add/writes its exclusive dQ. No cross-program dQ atomic is necessary. Source fixes eight warps for this historical backend; other native configurations are unverified. |
| flash_attn_triton.py:200-260 | Finite rescalable state concepts; log-normalizer reference recurrence gap | Forward stores natural LSE only. It chooses reference max(current tile max,previous LSE), not necessarily the running maximum or a bounded-lag maximum. It rescales the old raw numerator, adds current PV and updates LSE=reference+log(exp(old_LSE-reference)+new_mass), then finalizes with exp(reference-LSE). Existing online_rescaled_reduction@2's reference-policy menu cannot supply this choice by name. |
| flash_attn_triton.py:378-395,600-690 | softmax_gradient_row_checkpoint@1 and softmax_probability_recomputation@1 finite-row binding | Preprocess saves dot(O,dO). Recompute scores with original scale/bias and P=exp(score-LSE); form P*(dP-D) for derivative contractions. Vector or matrix bias, arbitrary row/key/head tails and causal predicates must match forward. The wrapper does not implement a general bias gradient. |
| flash_attn_triton.py:721-805,895-1005 | Output ownership alternative; scalar additive combine binding | Sequential key mode gives one batch/head program exclusive dQ; sequence-parallel mode launches key-tile programs and atomically sums dQ, while dK/dV stay key-owned. This changes parallelism and numerical addition order without changing the derivative contributor set. |
| Both forward files' immediate temporary store/load comments | Historical compiler realization gap | A scratch store/load is used to force a compiler behavior around a scale value. It is not a mathematical producer/consumer dependency or measured reusable optimization. No attempt is made to repair or benchmark the historical compiler. |

For raw frame weights(1,3), denominator l=4, scaled dO'=dO/4 yields
P'*dO'=(1/4,3/4)*dO. Omitting either l in preprocess or retaining it twice
changes the gradients. For the later recurrence, four equal old logits0 give
old LSE=log4 although running max remains0. The next tile may therefore use
reference log4 rather than0. A correct rescaled numerator still represents the
same sum, but the reference is outside a claim that it is the running max.
These are explicit representation/policy gaps, not missing source coverage.
