# Quantization, norm and activation design inventory

Paths are relative to the retained [tree](inputs/sources.json). Group by distinct
numerical contract, layout or dependency organization; backend menus do not
multiply the number of designs. Current [representative trace](trace-standalone-quantization-and-norm.md)
records concrete reuse and the replicated-cluster candidate.

| Design family | Entry / implementation representatives | Disposition |
|---|---|---|
| MXFP8/MXFP4 ceiling quantization | quantization/{fp8_quantization,fp4_quantization}.py -> kernels/{mxfp8_quantize,mxfp4_quantize}.py, quantization_cute_dsl_utils.py; grouped cuTile | Ceiling MXFP8 normal-range reuse, MXFP4 format/approximate divisor and byte0 payload-erasure boundaries recorded. Additional input dtype, layout and native conversion menus remain. Keep separate from floor-anchor GEMM output. |
| NVFP4 hierarchical scaling | quantization/fp4_quantization.py -> kernels/nvfp4_quantize.py, nvfp4_quantization_utils.py and csrc CUDA; fp4_layout.cuh | Coarse global/local rounded-scale composition exists in hierarchical_low_float_scaling@1; [encoding/decode direction and map4/map6 selector fields bound](trace-quantization-recipes-and-segmented-masks.md). Per-token scale coherence and native zero/range obligations remain explicit. |
| Scale layout and grouped storage | block_scale_interleave, shuffle_matrix_sf_a, grouped quantization ->128x4/8x4/linear kernels and fp4_layout.cuh |128x4 scatter/inverse coordinates checked. Grouped cuTile distinguishes payload expert offsets and padded metadata offsets, with a device-built tile prefix and fresh binary lookup. [8x4 map recorded](trace-quantization-recipes-and-segmented-masks.md); transpose and physical padding variants remain. |
| Per-token group8-bit | per_token_group_quant_8bit -> include/flashinfer/quantization.cuh and cutile counterpart | Correctly rounded FP32 division versus power-of-two reciprocal branch recorded. Output metadata stride/orientation, static scale floors and dispatch shapes remain native binding work. |
| Bit packing | quantization/packbits.py -> quantization.cuh | [segmented_bit_packing@1 extracted](trace-quantization-recipes-and-segmented-masks.md): both orders, independent byte prefixes, empty segments and zero short tails;2048 logical cases pass. Native CUB and prefix publication remain obligations. |
| KV quantize/dequantize/materialize | fp4_quantization.py:nvfp4_*kv*, kernels/nvfp4_kv_materialize.py, rope_quantize_fp8_cutile.py | Relate to existing cache record/generation and decode materialization traces; standalone endpoint scale orientation, page mapping and publication remain. |
| Classic and fused RMSNorm | norm/{__init__,kernels/rmsnorm,kernels/fused_add_rmsnorm}.py, include/flashinfer/norm.cuh | One-row statistics, vector-tail legality, input reload versus FP32 shared residual and mandatory T checkpoint recorded. Additional CuTe persistent/two-pass layouts remain named. |
| LayerNorm and adaptive norm | norm/kernels/layernorm.py, norm.cuh, norm/fused_dit_layernorm.cuh, norm/ln_fwd_silu_kernel.cuh | Raw versus centered statistic and SMEM caching choices recorded. Adaptive shifts/scales, optional quantized reductions and SiLU native variants remain. |
| Gated residual and next norm | norm/kernels/{gated_residual_rmsnorm,gated_residual_next_rmsnorm,rmsnorm_sigmoid_gate}.py | Full-row recomputation per output shard and8-CTA replicated partials are distinguished. Two dependent rounded-residual statistics and retirement recorded; [post-statistic sigmoid placement recorded](trace-quantization-recipes-and-segmented-masks.md); remaining native input-precision variants named. |
| Fused FP4 norm | cute_dsl/{rmsnorm_fp4quant,add_rmsnorm_fp4quant}.py | FP32 statistic input versus rounded residual quantizer input and cluster write/reread publication recorded. Complete NVFP4/MXFP4 scale/global-decoder composition and alternate layouts remain. |
| QK norm/rotary and concatenation/packing | norm/fused_qk_rmsnorm_rope.cuh, concat_ops.py, mla_kv_pack.py and corresponding include files | [Pair supply, across-head QK statistics, checkpoint, cache identity and MLA pack traced](trace-rotary-norm-and-cache.md); remaining native/model differences in [rotary inventory](rotary-cache-inventory.md). |
| Standalone/fused gates | activation.py, activation.cuh, gated_act_mxfp8.py and fp4/fp8 fused silu wrappers | Local act-and-multiply is an elementwise fusion; dtype checkpoints and quantization domain must remain separate. [Split vector indexing, unrolled T staging and FP32 activation/multiply recorded](trace-quantization-recipes-and-segmented-masks.md); backward relatives remain. |

This is an active auxiliary-family inventory. It does not close FlashInfer or
claim native execution. Continue by the outstanding design differences above,
then reconcile the full-library [inventory](inventory.md).
