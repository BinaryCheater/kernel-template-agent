# NVIDIA design inventory and reconciliation

Pinned [inputs](inputs/sources.json). All distinct groups have dispositions. Paths below
are relative to the retained tree. Compare seeded FA2/FA3 rules and later study
proposals by actual fields and scope, not mechanism names alone.

| Group | Entries / representative | Differences to account for | Status |
|---|---|---|---|
| CuTe forward Ampere/Ada | `cute/interface.py:1188`, `cute/flash_fwd.py` under `flash_attn/` | Register MMA, online state, staging and tails | [Traced by difference](trace-cute-forward-variants.md) |
| CuTe forward Hopper | `cute/interface.py:1206`, `cute/flash_fwd_sm90.py` | Warpgroup stages, masked online state, producer/consumer ownership | [Dense representative](trace-cute-sm90-forward.md), [indexing](trace-scheduler-and-indexing.md) and [sparse relatives](trace-sparsity.md) traced |
| CuTe forward Blackwell | `cute/interface.py:1301`, `cute/flash_fwd_sm100.py` | TMEM, 1/2-CTA MMA, softmax/MMA overlap, FP8 descales and score modifications | [Representative](trace-cute-sm100-forward.md), [variants](trace-cute-forward-variants.md), [indexing](trace-scheduler-and-indexing.md) and [sparsity](trace-sparsity.md) traced |
| Decode and operand swap | `cute/flash_fwd_decode_sm100.py` | SwapAB ownership and split-K/decode scheduling | [Traced](trace-swapab-decode.md); shared indexing/split restrictions recorded in their groups |
| CuTe forward SM120 | `cute/interface.py:1324`, `cute/flash_fwd_sm120.py` | Available ISA versus inherited pipeline, architecture-specific legality | [Traced by difference](trace-cute-forward-variants.md) |
| Sparse/MLA forward | `cute/flash_fwd_mla{,_1cta,_1cta_kb64}_sm100.py`, `cute/topk_gather_kv.py` | Absorbed large-value dimension, query-head packing, gathered support, 1/2-CTA regimes | [Traced](trace-mla-forward.md); native mapping/readiness gaps named |
| CuTe backward Ampere/Hopper/SM120 | `cute/flash_bwd{,_sm90,_sm120}.py` | Recompute probability, derivative reductions, atomic/deterministic output ownership | [Representative and differences traced](trace-cute-backward.md); ordered bulk-add realization/rank gap named |
| CuTe backward Blackwell | `cute/flash_bwd_sm100.py`, `cute/sm100_hd256_2cta_fmha_backward*.py` | dQ/dK/dV schedules, large-head specialized decomposition, TMEM and cluster ownership | [Main differences](trace-cute-backward.md) and [specialized head256](trace-specialized-backward-and-combine.md) traced; ownership/recomputation graph and native layout gaps named |
| Sparse/MLA backward | `cute/flash_bwd_mla*.py` | Sparse attention derivative plus dQ/dQv/dK GEMMs, head64 difference | [Traced](trace-specialized-backward-and-combine.md); indexed scatter, saved weights and bounded transient extracted |
| Pre/postprocess and split combine | `cute/flash_bwd_{preprocess,postprocess}.py`, `flash_fwd_combine.py`, `bwd_accum_tile.py` | Delta/LSE, layout/cast accumulation, split softmax merge and deterministic order | [Delta/LSE, residual and sink derivative](trace-cute-backward.md), [combine and accumulator layout](trace-specialized-backward-and-combine.md) traced; native coordinate/layout gaps named |
| Mask, sparsity and scheduling | `cute/{softmax,mask,block_info,block_sparsity,compute_block_sparsity,cu_blocks_kernel,tile_scheduler,prepare_scheduler,pack_gqa,paged_kv}.py` | Full/partial/empty tiles, sparse metadata, variable lengths, paging, grouped heads, persistent assignment | [Indexing/scheduling](trace-scheduler-and-indexing.md) and [sparsity](trace-sparsity.md) traced; native claim/classification gaps named |
| Configuration, pipelines and instruction helpers | `cute/{config,autotune,pipeline,barrier,named_barrier,copy_utils,ampere_helpers,blackwell_helpers,mma_sm100_desc,tensormap_utils,fast_math,utils}.py` | Compiled menu vs heuristic, transaction protocols, descriptor mutation, exact mappings/math | Bound through the forward/backward consumer traces; native inline waits, descriptor lowering and math/layout obligations named there |
| CUDA FA2 | `flash_attn/flash_attn_interface.py`, `csrc/flash_attn/`, root `setup.py` | Training/inference, split KV, cache append, rotary, dropout, local/softcap/ALiBi variants | [Traced](trace-cuda-fa2.md); random replay/sign encoding extracted, mutating-cache and numeric binding gaps named |
| CUDA Hopper FA3 | `hopper/flash_attn_interface.py`, `hopper/flash_api.cpp`, forward/backward mainloops and launchers | FP8 layouts, paged/cache/varlen, persistent schedule, backward and split-combine differences | [Traced](trace-cuda-fa3.md); paired FP8 permutation, ordered-add and native schedule/layout gaps named |
| Legacy NVIDIA Triton attention | `flash_attn/flash_attn_triton.py`, `flash_attn/flash_attn_triton_og.py` | Earlier bias/mask, saved normalizer/output, recomputed backward and key/query ownership variants | [Traced](trace-legacy-triton-attention.md); recurrence/representation gaps named |
| Auxiliary NVIDIA operators | `csrc/fused_dense_lib`, `csrc/layer_norm`, `flash_attn/ops/triton`, Python ops/layers/losses/modules | Dense/activation/bias gradients, dropout/residual/norm, rotary and cross-entropy families | [Traced](trace-auxiliary-operators.md); retention and vocabulary loss extracted, provider/map gaps named |

Final groups were reconciled against Python entry/build registrations, generated
CUDA instantiations, header/device locations and public auxiliary operators.
Tests, benchmarks and examples are caller evidence; AMD CK/Triton backends under
ROCm-only paths and external submodule device bodies remain excluded. The full
library scope is not limited to the catalog's two forward files.

## Auxiliary and legacy reconciliation

Top-level csrc contains the three studied attention backends/dependencies plus
fused_dense_lib and layer_norm. The NVIDIA groups below are accounted for; Python
modules that compose them are callers, not automatically separate native designs.
Their [device trace](trace-auxiliary-operators.md) and
[legacy attention trace](trace-legacy-triton-attention.md) give dispositions for
all rows below. No installation was needed.

| Group | Caller / registration -> device representative | Distinct choices to settle |
|---|---|---|
| Earlier Triton attention | flash_attn/flash_attn_triton{,_og}.py autograd entry -> _fwd_kernel/_bwd_kernel | Bias/layout tails, normalizer convention, scratch, recompute and gradient ownership; compare with already extracted attention rules |
| cuBLASLt fused dense / MLP | csrc/fused_dense_lib/setup.py -> fused_dense.cpp / fused_dense_cuda.cu; flash_attn/ops/fused_dense.py | Bias/GELU epilogues and AUX checkpoint, dgrad/wgrad/bias gradients, heuristic/workspace, activation checkpoint/recompute |
| Triton dense / activation / MLP | flash_attn/ops/triton/{linear,k_activations,mlp}.py | Split-K and atomic initialization, fused activation/input-grad epilogues, save/recompute choices and legal shapes |
| CUDA dropout + residual + LN/RMSNorm | csrc/layer_norm/setup.py -> ln_api.cpp -> ln_{fwd,bwd,parallel_residual_*}_kernels.cuh | Standard and parallel residual, subset row maps, dropout state, normalization reduction and affine gradient finalizer |
| Triton LN/RMSNorm / norm+linear | flash_attn/ops/triton/layer_norm.py -> _layer_norm_fwd_1pass_kernel/_layer_norm_bwd_kernel / LayerNormLinearFn | Row statistics, residual and parallel inputs, dropout, partial affine reductions, recomputation/linear boundary |
| Rotary helper | flash_attn/layers/rotary.py -> flash_attn/ops/triton/rotary.py | Interleaved/half-split pairs, in-place/backward conjugate, packed QKV and ragged sequence position mapping |
| Cross entropy | flash_attn/losses/cross_entropy.py -> flash_attn/ops/triton/cross_entropy.py | Vocabulary split/distributed normalization, smoothing, ignored labels, logit scaling and z-loss, gradient ownership |
| Composition / distributed wrappers | flash_attn/modules/{mha,mlp,block,embedding}.py, flash_attn/utils/distributed.py | Determine which calls only compose the listed kernels and which introduce communication/chunk/checkpoint choices needing a source-backed disposition |

The layer_norm README labels its CUDA extension historical, with Triton used
by current modules. Historical retained NVIDIA kernels still belong to this
library assignment; that label is not a reason to omit their distinct designs.

Closure: all retained NVIDIA device families, generated specializations and
public operator groups have design dispositions. External CUTLASS/CK/aiter
submodules were not materialized, AMD paths are excluded, and provider kernels
are boundaries rather than invented device implementations. The study closes
with fifteen provisional pairs, retained logical checks and named extraction/
realization gaps. Native replay, performance and independent admission are
separate work; a WP gate/commit was not part of this source-study assignment.
