# SVDQuant scale placement and native W4A16 decoder binding

Paths refer to the retained [tree](inputs/sources.json). This pass follows
`flashinfer/gemm/{gemm_svdquant,cake_quant_lora_gemm,gemm_bf16_fp4}.py` into
retained generated CUDA, the B12x fused epilogue and native BF16×FP4 kernels.
No absent dependency or separate paused CUTLASS source study is resumed.

## Prepared low-rank factors change both work and rounding

Retrieval cue: repeated quantized-base GEMMs reuse the same BF16 adapter and
base scale. New [low_rank_scale_placement@1](delta/low_rank_scale_placement@1/template.yaml)
and [proof](delta/low_rank_scale_placement@1/proof.yaml) expose two choices:

| Choice | Arithmetic | Boundary |
|---|---|---|
| runtime_dual_scale | alpha*R + lambda*(D*U^T) + bias | Raw BF16 factor, independent runtime scales; alpha0 can retain a nonzero correction |
| prepared_up_fold | Uf=BF16(FP32(U)*RN_FP32(lambda/alpha)); alpha*(R+D*Uf^T)+bias | Nonzero alpha, admitted new BF16 factor rounding, immutable parameter snapshot and ordered refold |

R is the residual base product in the selected kernel units; D is the declared
down result. The real nonzero-alpha identity does not prove bitwise equality
after Uf re-rounding. Bias remains outside alpha. Output/correction accumulation
order and the zero-adapter branch are explicit native numeric bindings.

| Source choice | Candidate field / obligation | Evidence |
|---|---|---|
| `cake_quant_lora_gemm.py:14-39,759-820` | `placement`, `snapshot_coherence` | Raw path consumes original BF16 factors and independent scalars. Prepared adapter retains U/lambda/alpha source references plus pointer/version/shape/device pins. |
| Same file :825-945 | `fold_error`, `pre_fold` | Initial preparation checks alpha!=0 and finite ratio; in-place preparation computes device FP32 ratio/product and one BF16 copy. Alpha is the prepared object's same scalar binding. Refold after changes; equal shape/pointer alone is insufficient. |
| Same file :759-801,939 onward | `pre_snapshot`, `pre_lifetime` | Framework versions detect tracked writes; arbitrary device/external writes require explicit refold. A graph replay cannot be assumed to rerun host freshness checks. Ordered graph refold publishes new Uf before its GEMM and must not overlap old readers. |
| `jit/cake_quant_lora_gemm.py:771-787` -> `csrc/cake_quant_lora_gemm/cake_quant_lora_gemm_140e35bb5ecf92090cf9_kernel.cu:952-974,1104-1120` | Prepared native representative | Source registers ranks32/64/128 for the prepared NVFP4 program. It chooses adapter-off from the scalar, uses common alpha and applies bias in the final FP32 result before BF16 store. FP8 raw support does not imply a prepared FP8 mode. |
| `gemm_svdquant.py:792-865` | Common-scale correction | Caller supplies L1/alpha for the fused residual-plus-low-rank epilogue. Backend/rank menus remain distinct; one native API name does not erase factor-preparation rounding. |

For exact real arithmetic the factor-rounding contribution satisfies the
per-output bound |alpha|*sum_r |D_r|*|Uf_r-(lambda/alpha)*U_r|. Native
FP32 ratio/product/MMA/output rounding adds its own error terms. The
[logical check](checks/check_low_rank_scale_placement.json) uses alpha3,
lambda1, BF16 factors and a dyadic down vector to show a fold error visible
after final BF16 output; a shifted bias moves the result into a rounding bin
that hides the same error. No universal bitwise difference or equality follows.

## Smoothing has two separately rounded input channels

`gemm_svdquant.py:1076-1198` describes the complete linear chain:

- Residual branch: quantize BF16(x*pre_quant_scale) to NVFP4 with its declared group/global scales.
- Down branch: D=BF16(x*L2T_smoothed), where L2T_smoothed=BF16(pre_quant_scale[:,None]*L2^T) was prepared earlier.
- Up/correction: use L1_scaled=BF16(L1/alpha) in the common-scale fused epilogue.

The down branch consumes original x, not the quantized residual activation.
Applying smoothing again to x before multiplying L2T_smoothed doubles it.
The real diagonal-factor identity does not remove the BF16 smoothing, prepared
factor and down-output checkpoints. The logical check distinguishes all three
input choices on identifiable two-channel vectors.

`include/flashinfer/gemm/nvfp4_smooth_quantize_sm100.cuh:338-365,499-503`
applies BF16 pair multiplication before quantizer statistics/encoding. Existing
hierarchical_low_float_scaling@1/groupwise decoder concepts apply only with the
source global/local scale/cast order bound; moving smoothing through the encoder
changes its codebook inputs. The optional fused smooth+down SM120 header keeps
these as separate operations/roles rather than replacing D by a decoded NVFP4
activation.

## Fused correction and unfused comparison have different checkpoints

`gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py:1836-1877` publishes BF16
rank tiles in released A/B mainloop storage, accumulates each rank tile directly
into the FP32 residual accumulator, and preserves C-fragment ownership when
swap_ab swaps correction operands too. :1909-1911 applies alpha then bias before
output conversion. operator_fusion@2 supplies the local dependency/lifetime
requirements; the new scale-placement rule supplies factor/scalar semantics.
Reusing mainloop storage requires all old operand readers to retire first.

By contrast, `gemm_svdquant.py:318-357` writes the residual GEMM to BF16 out,
computes a BF16 correction tensor, scales it in place, adds into BF16 out and
then adds bias. Its multiple output checkpoints are a compositional comparison
policy, not the same single-final-cast fused schedule. The logical witness
BF16(1.003-0.999) versus BF16(BF16(1.003)+BF16(-0.999)) distinguishes them.
Any differential tolerance must account for that numeric difference.

Native preprocessing/correction overlap and PDL publication remain source-bound
protocol obligations; a quantized-residual-ready event does not automatically
certify a completed down result. No additional GPU replay is needed to identify
that dependency.

## BF16×FP4: preserve the decoder while changing placement

`gemm/kernels/native_bf16_fp4/README.md` and `support.py` define an explicit
SM120/121 canonical-weight path: packed E2M1[N,K/2], padded128x4 E4M3 scales,
K divisible16, BF16 A and BF16/FP16 output. It does not prepare an alternate
full weight matrix; other public backends may consume other representations.

| Source choice | Existing rule / field | Binding |
|---|---|---|
| `kernel.py:27-61` | decoded_tile_materialization@1 decoder identity | CUDA13.2+ converts E2M1 pairs and E4M3-derived scales directly to BF16 then multiplies BF16. Older path decodes through FP16/FP32 then casts the scaled result to BF16. E4M3 scale values and E2M1 values embed exactly in BF16, so the finite decoder identity is checkable. It does not narrow BF16 A to FP16. |
| `staged_kernel.py:156-163,240-247` and scalar/native kernels | Decoder placement/tiling | Stage packed weights/scales, then decode fragments with the matching shared decoder. A whole decoded-tile materialization is legal only if it preserves that exact decoder and fits; the source staging of packed bytes is not automatically a decoded-tile instance. |
| `kernel.py:230-254` | reduction_partition@1 and explicit linear partial representation | Split-K stores FP32 product partials, reduces split slots in fixed order, applies optional alpha once, then casts output. These are linear GEMM partials, not normalized attention O/LSE states. |
| `support.py:11-45` | Packed representation and address guard | Positive N/K, K%16=0, compact scale byte count,4-byte weight alignment and index products below2^31. Ragged N/M tails retain their selected kernel predicates. |

The logical check exhausts1905 finite E2M1/E4M3 scale combinations for the
BF16 multiply versus FP32-product-then-BF16 decoder identity, in addition to
scale folding, stale parameters, alpha0, bias placement and smoothing witnesses.
Native packed conversion, FTZ/fast math, asynchronous staging, split reduction
association and publication remain explicit realization obligations.

Next: task-scheduled routed GEMM operands, remaining low-bit output quantizers
and deterministic grouped training. Low-rank scale/smoothing and native-layout
W4A16 now have source-backed dispositions; other low-bit/provider recipes remain
in the GEMM/full-library inventory.
