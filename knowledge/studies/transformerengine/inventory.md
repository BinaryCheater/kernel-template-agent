# TransformerEngine design coverage

Scope: NVIDIA paths in the retained top-level tree. Public C entry points under
`transformer_engine/common/include/transformer_engine/`, common CMake source
registrations, framework dispatch and device bodies define the operator scope.
Python/PyTorch/JAX wrappers select these backends; external cuDNN/cuBLAS/NCCL
implementations remain external rather than being treated as available source.

| Design group | Representative routes under `transformer_engine/common/` | Disposition |
|---|---|---|
| Scale domain and update epoch | `recipe/current_scaling.cu`, `delayed_scaling.cu`, `multi_tensor/compute_scale.cu` | First comparison read: current whole-tensor two-pass dependency, delayed history, batched scale metadata; graph noop is observable state behavior. |
| Low-float representation and grouping | `cast/fp8/`, `fp8_blockwise/`, `mxfp8/`, `nvfp4/`, `recipe/`, `CuTeDSL/cast/`, `triton/` | Surveyed: tensor/row/block domains, independent orientations, MXFP8 register/TMA variants, nested NVFP4, 4over6 selection and grouped requantization. Three provisional pairs; exponent-generation/RNG/backend gaps named. |
| Transpose, cast and fused epilogues | `transpose/`, `cast/dispatch/`, `activation/`, `cast_grouped*`, `cast_dbias*` | Surveyed: same-code cast transpose versus independent quantization, padded shared relayout, activation/gradient/dbias fusion, grouped descriptor updates and inverse metadata. |
| Normalization | `normalization/layernorm/`, `rmsnorm/` | Surveyed: centered-moment merge, two-pass general path, reusable CTA-group allreduce, backward parameter partials, external cuDNN plans and split-statistics MXFP8 fusion. Two new pairs and one finite logical protocol check. |
| Softmax and fused attention | `fused_softmax/`, `fused_attn/`, `cudnn_utils*` | Surveyed: retained-row softmax/mask variants, external cuDNN SDPA/FP8 graph gates, natural-LSE context merge, dual-chunk permutation and paged cache append. Existing-rule reuse and native/numerical boundaries recorded. |
| Rotary, activation, dropout and transforms | `fused_rope/`, `activation/`, `dropout/`, `hadamard_transform/`, `swizzle/` | Surveyed: pair-address rotary maps, MLA rotate/quantize fusion, gated activation/scale gradients, logical-chunk Philox mask, matrix-based signed H16 and grouped transport. Numeric/RNG/partner-map boundaries named. |
| Routing and token movement | `fused_router/`, `permutation/` | Surveyed: score/selection distinction, grouped top-k, packed four-bit radix with tie policy, cutoff histograms, bitmap/indices and prefix route packing/output-owned combine. Radix version 2 proposed. |
| GEMM and communication overlap | `gemm/`, `comm_gemm/`, `comm_gemm_overlap/`, `ep/`, `nvshmem_api/` | Surveyed: cuBLASLt/batched/grouped and cuBLASMp/EP external boundaries, event/counter chunk handoff, userbuffer ring/RS publication, routed scale metadata and NVSHMEM wait epochs. One new handoff pair; native provider/progress obligations named. |
| Multi-tensor optimizer/statistics | `multi_tensor/{adam,sgd,l2norm,scale}.cu`, `multi_tensor_apply.cuh` | Surveyed: bounded tensor/chunk batches, Adam/SGD recipe ordering, master/model metadata and L2/statistic combine. New carry-aware master split with exhaustive low-word model check. |
| Matrix iteration and runtime selection | `newton_schulz/`, `util/rtc*`, CMake/dispatch and framework selection | cuSolverMp Newton–Schulz is an external device implementation; sharding, coefficient/workspace and stream contracts recorded. CMake/RTC menus followed through representative quantization/normalization callers; communication runtime survey remains with its row. |

All design groups have source-backed dispositions. Closure accounts for distinct
designs and named gaps rather than an entry-by-entry implementation audit.

Additional Triton operators: `cross_entropy.py` uses streamed and tensor-parallel
normalization state; `mhc.py` uses n=4 log-domain iterative mixing with stored
versus recomputed history; `pad.py` is guarded storage padding. Their dispositions
are in [routing/transforms/state](trace-routing-transforms-and-state.md).

Reconciliation: immediate common source groups (cast/recipe/transpose, normalization,
softmax/attention, transforms/activation/dropout/swizzle, router/permutation,
multi-tensor, GEMM/comm/EP/NVSHMEM and Newton–Schulz) all map to the rows above.
Common CMake source registrations use those same groups. Public C operation
headers add no unassigned family; include/cmake/util and Python/JAX/PyTorch entry
layers provide contracts, dispatch, code generation or wrapping. No CUDA device
files occur under transformer_engine outside common in this snapshot. External
submodules and provider device kernels are explicitly unavailable, not implicit
coverage. Source-design pass closed with eight provisional pairs.
