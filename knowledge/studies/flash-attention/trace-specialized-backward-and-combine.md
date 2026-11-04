# Specialized backward, indexed accumulation and bounded scratch

Source paths are relative to flash_attn/cute/ in the
[retained tree](inputs/sources.json). This continues the
[representative backward trace](trace-cute-backward.md), closing the distinct
CuTe backward and combine designs while naming native mapping gaps.

## Large-head ownership split

The dedicated head256 path avoids a cross-CTA dQ combine by giving one kernel
the complete key reduction for each query tile. A second kernel owns KV tiles
and reduces all contributing query tiles and grouped heads. The two kernels
repeat score/probability/derivative work but each writes its own output. This is
a decomposition choice, not a different softmax derivative.

| Source decision | Rule or gap | Concrete binding / obligation |
|---|---|---|
| sm100_hd256_2cta_fmha_backward.py:28-106,185-231 | Output-ownership/recomputation split gap; reduction_partition@1 describes each already-defined reduction | Fixed Q/K/value dimension256; dQ tile128x128, dKdV128x64. Launch dQ first, then dKdV. The wrapper rejects local attention, custom scores/masks, sparse blocks and semaphore arrays. This path cannot be silently substituted for the general feature-rich backward. |
| sm100_hd256_2cta_fmha_backward_dqkernel.py:1264-1858,1942-1968 | Existing probability/row-center rules; output_tiling@2 and compute_instruction_selection@1 tcgen05, with twoCTA realization gap | A query-owned tile visits its admissible KV interval; the first dS*K initializes dQ and later key tiles accumulate it. QK(next), dP and dQ steps are interleaved with explicit pipeline handles. Each operand generation releases after its final dependent MMA, not simply after a pointwise P becomes ready. |
| dqkernel.py:2136-2253,2259-2386 under the same prefix | softmax_probability_recomputation@1 plus softmax_gradient_row_checkpoint@1; physical_storage@1 | Reconstruct P in FP32 from scaled score/base2 LSE, form dS with the row center and score scale once, then cast for dQ. TMEM dQ drains to dedicated output ownership. Aligned tiles use shared/TMA stores; query tails use predicated direct stores. Empty valid query work gets explicit zero stores. |
| sm100_hd256_2cta_fmha_backward_dkdvkernel.py:1227-1253,1858-1885,2073-2077 | reduction_partition@1 over query-block x within-KV-group query-head contributors | iter_count=(query_end-query_start)*query_heads_per_KV. On reaching query_end, reset query index and advance the grouped query head. Thus a GQA head contribution is not replaced by a final modulo view. Physical cu-sequence offsets stay separate from effective seqused lengths. Empty valid KV tiles explicitly clear dK/dV. |
| dkdvkernel.py:2668-2785,3100-3179 under the same prefix | Same normalized-P and row-center choices; final scale/cast ownership | dV accumulates P^T*dO, dK accumulates dS^T*Q. Causal dS is explicitly zeroed before conversion. dK final output applies the remaining softmax scale, while dV does not. Epilogues cast before predicated/TMA stores. No semaphore/atomic dQ reduction is needed in this decomposition. |

For Hq/Hkv=3 and two admissible query blocks, one KV output tile has six
logical query-block/head contributors. Reducing only the two blocks for head0
loses heads1/2 even though the physical K/V view is shared. General backward
instead uses separate head contributors and a combine; both must represent
the same complete contributor set. A reusable ownership/recomputation split
needs fields for repeated intermediate producers and each output's reduction
domain; current reduction_partition alone does not choose that graph.

## Sparse MLA derivative decomposition

For absorbed scores Z=Q*K_rope^T+Qv*V^T, V has two derivative paths:
P^T*dO from the weighted output and dZ^T*Qv from its score role. Their sum
must reach the same original V row. The rope K path is dZ^T*Q. The main
backward writes a dS/dZ transient, while gather GEMMs produce dQ=dS*K_rope
and dQv=dS*V. The source dS includes the score scale for these later GEMMs.

| Source decision | Rule / field or gap | Concrete binding / boundary |
|---|---|---|
| flash_fwd_mla_sm100.py:2897-2934; flash_bwd_preprocess.py:550-580; flash_bwd_mla_sm100.py:2906-2940 | [shifted_weight_checkpoint@1](delta/shifted_weight_checkpoint@1/template.yaml) | TwoCTA forward saves converted unnormalized P and the actual running reference per row/tile. Preprocess forms exp2(scale_log2*reference-LSElog2). Backward decodes BF16 weights into FP32 for dS and rounds separately for dV. A running reference is not necessarily a local tile maximum. The softmax helper uses finite reference0 for the first all-masked tile; its defined zero weights can decode through that finite frame. OneCTA forward variants reject mP/mRowMax. |
| flash_bwd_mla_sm100.py:2570-2728,2941-3068 | Existing probability recomputation for finite rows; explicit masked/empty binding | Recompute QvV plus optional rope QK before one normalization. Invalid gathered rows get exponent -inf; invalid dP is forced zero before the P*dP product. Retain FP32 P for pointwise work and convert separately for MMA. Saved-P and recompute-P have different checkpoint error/traffic. |
| flash_bwd_mla_sm100.py:3270-3368 | [indexed_additive_scatter@1](delta/indexed_additive_scatter@1/template.yaml), four_elements | dV score/output paths and optional fused dK_rope use original top-k indices, with lower/upper logical bounds and four FP32 atomic adds. Every duplicate slot remains a separate derivative contribution. Range checks matter when a chunk shrinks the main kernel's K view. |
| flash_bwd_mla_dk_sm100.py:70-110,1005-1134 | indexed_additive_scatter@1 scalar; compute_instruction_selection@1 tcgen05 and physical_storage@1 | Generic separate dK contracts gathered-slot dS^T with Q over heads, then scatters slot results back. Fixed BF16 inputs/FP32 output. TMEM drains through shared staging before scalar atomics. Local code guards negative sentinels; upper bounds must come from the caller. The unusual topk%256==1 twoCTA constructor branch is a source menu, not a derived necessity or an admitted caller case. |
| flash_bwd_mla_dq_dqv_sm100.py:1-33,57-88,809-837 | indexed_gather@1 registers plus output_tiling@2; cluster-native mapping gap | General path uses query-head tile128 and two CTAs across value output halves; each uses oneCTA MMA, dS is multicast, and only CTA0 produces the rope dQ. Reuse the original per-token top-k list. This is not twoCTA MMA merely because two CTAs cooperate. |
| flash_bwd_mla_dq_dqv_sm100_h64.py:60-91,295-314,845-978 | grouped_head_row_packing@1 for logical identity only; guarded head padding and M64 mapping separate | One token/CTA,1..64 real heads padded to64, latent512 split into two256 outputs and optional rope64. Gather64 keys per stage once, reuse the same swizzled bytes through an MN-major MMA view. Shared swizzle equality is asserted. Every dS/KV generation is read by all relevant GEMMs before release. |
| h64.py:845-873 under the same prefix | indexed_gather@1 index_prefetch registers,index_reuse1; multi_buffer_rotation@2 once generation bound | Two register sets prefetch indices n+2 while copying n. Top-k is a positive multiple128, so64-key stages come in complete pairs. KV stage count and index-buffer parity are distinct. Dedicated epilogue staging lets the next token's gather proceed without waiting for output stores on an aliased KV buffer. |
| interface.py:2763-2775 | Explicit compiled/dispatch feature menu | MQA only; top-k divisible128; deterministic backward rejected. Recompute mode, real Q/K and head tile64 enable fused rope-dK in the main kernel; otherwise launch separate dK. Both modes must sum the same derivatives. Native value-layout/precision and fused epilogue schedule remain implementation obligations. |

The two V paths follow the chain rule for a shared operand. With scalar V,
Qv=1, logits=V, dO=1, V=(0,ln3), output O=(3/4)*ln3 and P=(1/4,3/4),
the V gradient is P + P*(V-O). Keeping only P^T*dO loses the score path;
keeping only dS^T*Qv loses the direct output path. Both are accumulated
through the same original index map. This algebra does not prove native
mixed-precision execution, which remains explicitly bound.

## Reuse one chunk transient

interface.py:2823-2859,3000-3110 implements
[consumer_complete_scratch_chunking@1](delta/consumer_complete_scratch_chunking@1/template.yaml).
The consumer set is main backward -> dQ/dQv -> optional separate dK. In fused
rope mode the main backward already owns dK, but dQ/dQv still consumes dS.
The host loop reuses dS only after these dependent operations in stream order.
Caller-provided dK/dV accumulators are retained; newly allocated ones are zeroed
outside the loop. dQ/dQv write original token slices. The last chunk has its
actual smaller extent. Source contiguous slicing admits varlen or batch1;
other batched layouts fall back to the unchunked path, not an invented copy.

The causal mask cannot use chunk-local query length with unchanged K length.
For Q5,K7,chunk[1,3), local row0 originally admits keys<=3. Naive local
Q2,K7 admits keys<=5. Source shrinks the main K view to K-Q+chunk_end=5,
recovering limit3 while the subsequent gather GEMMs retain original K storage.
For varlen, clipped query offsets and effective K ends are valid only because
all earlier documents retain their full key spans and later documents have no
query rows in the chunk. This is a binding invariant, not general permission
to treat modified cumulative ends as original starts. Fully masked chunks
write zero dQ/dQv and omit additive K/V work.

The [mapping check](checks/check_sparse_backward_mapping.py) covers duplicate
scatter, invalid lower/upper indices, one initialization across chunks,
tile-specific exponent references, two-stage BF16 rounding, causal coordinate
rebasing and complete tail/consumer traversal. [Results](checks/check_sparse_backward_mapping.json)
record the tested finite logical cases; asynchronous completion and varlen
native mapping remain implementation obligations.

## Split combine and accumulator layout

flash_fwd_combine.py:552-680 binds split_combine_kernel@1 mode rescaled_merge:
each split supplies normalized O_s and natural LSE_s. Compute m=max(LSE_s),
alpha_s=exp(LSE_s-m)/sum exp(LSE_s-m), and O=sum alpha_s*O_s. This differs
from merging unnormalized raw online states unless their representation is
decoded first. Dynamic split counts exclude inactive scratch; one dynamic
split still copies its partial to final output. Empty partials use LSE=-inf.
All-empty rows yield zero O and LSE=-inf, binding empty_support_normalization@1.
The O combine only multiplies when alpha>0, so an excluded partial's undefined
payload does not turn zero*NaN into NaN. Max-valid-split pruning skips trailing
identities, not arbitrary nonidentity contributions. cp.async stages form a
uniform ring; the source explicitly relies on each thread reading only its
own loaded shared data rather than a cross-thread publication barrier.

bwd_accum_tile.py:17-49 defines the flat FP32 scratch representation shared by
writer and postprocess. For oneCTA with R rows, offset(r,c)=4*r+(c modulo4)
+4*R*(c//4). For paired128 rows, let u=r modulo64, v=r//64, b=c//(D/2),
and j=c modulo(D/2); offset=4*u+v*64*D+(j modulo4)+512*(j//4)+256*b.
At R128,D64, use coordinate(r65,c5): oneCTA773, twoCTA4613. The row-major offset4165 is
different from both. The source's producer checks and postprocess compile key
carry the selected order. Existing physical_storage@1 gives lifetime but not
this coordinate map, and rank-three tensor_axis_permutation@1 cannot encode
the split/interleaved axes. Keep the TMEM-fragment scratch-layout binding gap
explicit; do not claim plain row-major scratch.

## Extraction review

Three independent choices produce three objects: collision accumulation,
persistent exponent tiles, and bounded multi-consumer scratch. Keeping only
the existing gather/live-state/fusion names would omit their index multiplicity,
per-tile frame and cross-kernel lifetime contracts. Each candidate has an
executable scope guard and paired outline. Their selectors bind atomic width,
actual saved frames and chunk rows respectively. The retained witnesses expose
the wrong alternatives: assignment at collisions, final-reference substitution,
and local-length causal rebasing. These source-backed choices close the CuTe
backward families with native atomic, cluster layout, schedule and numerical
obligations named, not with performance or runtime admission claims.
