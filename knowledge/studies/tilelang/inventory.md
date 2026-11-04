# TileLang design coverage

The retained top-level tree contains examples, Python language/layout/scheduling
interfaces, native lowering and tests. This inventory groups designs rather than
counting each variant as a separate study. AMD/Ascend backends are excluded;
other backend-specific examples are selected only on their NVIDIA path.

| Design group | Representative routes | Disposition |
|---|---|---|
| Dense tensor-core GEMM and cooperative persistence | `examples/gemm_tcgen05/`, `gemm/`, `gemm_sm120/`, `warp_specialize/` | Persistent representative read; base roles/stages reuse existing rules. Two-CTA resource unit and explicit output transfer remain bindings. |
| Tail-wave Stream-K and split reduction | `gemm_tcgen05/gemm_tcgen5mma_ws_persistent_streamk.py`, `gemm_streamk/`, `gemm_splitk/` | Representative read; balanced logical intervals, publish-before-fixup and final-owner gather recorded. |
| Low precision, quantization and sparse MMA | `blockscaled_gemm_sm100/`, `gemm_fp8/`, `gemm_int4/`, `dequantize_gemm/`, `bitnet-1.58b/`, `gemm_sp/`, `blocksparse_gemm/` | Surveyed: native scale transport, periodic accumulator promotion, packed integer/floating decode, 2:4 compression and block masks. One provisional compression pair; other choices reuse rules or retain explicit format bindings. |
| Grouped/routed GEMM | `grouped_gemm/`, `fusedmoe/`, `deepseek_deepgemm/` | Surveyed: prefix-decoded group work, cooperative padded peers, expert packing and weighted scatter reduction reuse existing scheduling/storage rules. |
| Dense/paged attention and sinks | `flash_attention/`, `flash_attention_sm100/`, `flash_decoding/`, `attention_sink/`, `deepseek_mla/` | Surveyed: online reduction, denominator-only sink, split merge, paged MLA, SM100 operand sites/roles and checkpoint-based backward recomputation. Native/cooperative-launch bindings retained. |
| Structural/data-selected sparse attention | `block_causal_attention/`, `blocksparse_attention/`, `deepseek_nsa/`, `deepseek_v32/`, `dsa_hisa/`, `dsa_sparse_finetune/`, `seer_attention/`, `minference/` | Surveyed: block masks, block/row/page gathers, hierarchical selection, mixed support and shared-frame output sharding. Dense equivalence, index multiplicity, reference policies and empty/tail cases are explicit boundaries. |
| Recurrent/linear attention and token mixing | `linear_attention/`, `gdn/`, `kda/`, `kpool/`, `deepseek_mhc/`, `deepseek_v4/` | Surveyed: chunked linear/retention/Mamba expansion, delta/WY state dependencies, KDA blocked inverse and mutable decode slots, pooled history and mHC mixing. Two new pairs; scan/WY producer and inter-chunk propagation gaps named. |
| Reductions, normalization and selection | `norm/`, `online_softmax/`, `topk/`, `deep_select/`, `gemv/`, `hadamard_transform/` | Surveyed: retained-row versus reload normalization, checkpoint softmax, repeated-max/radix/streaming/hierarchical top-k, GEMV ownership and multilevel Hadamard. Two new selection pairs; exceptional-value/width/normalization gaps named. |
| Convolution and pointwise/layout transforms | `convolution/`, `elementwise/`, `cast/`, `rand/` | Surveyed: implicit patch contraction and pointwise staging reuse; group-128 FP32-scale FP8 conversion needs a format variant; RNG stream mapping retained as a gap. |
| Lowering, layout inference and runtime selection | `tilelang/cuda/`, `tilelang/tileop/`, `src/`, `dynamic_shape/`, `eager_jit/`, `aws/`, `autodd/`, `iket/`, `analyze/`, layout examples | Surveyed: CUDA instruction/copy eligibility, coupled layout search, automatic/manual warp roles and independent persistent/input rings. Compiler-owned selections and reducer/scan lifecycle gaps recorded. No compiler installation performed. |

Shared mechanisms and test variants receive brief reuse dispositions. Source
availability alone does not establish native support; no GPU execution is claimed.

Reconciliation against the retained immediate `examples/` directories: `aws/`
is automatic CUDA warp specialization; `plot_layout/` and
`visual_layout_inference/` visualize layouts; `analyze/`, `autodd/`, `iket/` and
`eager_jit/` provide tooling/invocation examples. Root `quickstart.py` repeats
tiled GEMM. All other NVIDIA directories are assigned to the design rows above.
Backend tests and lowering registrations bind those designs rather than requiring
a separate trace per specialization. External submodule contents and unproved
native layouts remain named coverage limits.

Source-design pass closed. Evidence: [GEMM scheduling](trace-gemm-scheduling.md),
[representations](trace-gemm-representations.md), [attention](trace-attention.md),
[recurrence/mixing](trace-recurrence-and-mixing.md), and
[selection/lowering](trace-selection-and-lowering.md).
