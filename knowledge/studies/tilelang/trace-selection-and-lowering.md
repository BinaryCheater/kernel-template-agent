# Selection, small operators and compiler-owned choices

Paths are relative to the pinned retained source. This closes the remaining
design groups by reusable choice; source timing claims are not measurements here.

| Representative | Choice and pool disposition | Binding boundary |
|---|---|---|
| `examples/deep_select/example_deep_select.py:18–186,246–465` | New [radix_threshold_selection@1](delta/radix_threshold_selection@1/template.yaml) and [streaming_topk_filter@1](delta/streaming_topk_filter@1/template.yaml) | BF16/FP32 monotone keys, no NaNs, exact **unsorted** top-k with arbitrary tied identities. Byte histograms locate rank; greater/equal quotas produce exactly k. Streaming retains winners, filters against a conservative threshold and reserves `2*k+block` capacity before an all-hit tile. Source block permutation is bijective because its stride is coprime to block count. |
| Same file `prepare_deep_select:485–635`; `deep_select_config.py` | Hierarchical tiles produce local top-k, then repeated exact top-k of the union; source also offers streaming, splits, merge fan-in, vector/TMA load choices | Local-top-k union contains a global top-k, but `sorted_run_topk@1` does not supply an unsorted radix merge realization. Config uses dtype/k tier/length/batch/SM/shared-memory menus, not sampled timing. TMA path requires aligned full tiles and resource fit. Prepared runner reuses workspace and is not reentrant. |
| `examples/deepseek_v32/topk_selector.py:11–189` | Same radix rank idea with a coarse FP16 high-byte pass, then full FP32-byte refinement of the boundary bucket | A coarse bucket is not a final exact order. Worst-case boundary workspace is the entire row. Tie quota, count visibility and valid start/end range are required. |
| `examples/topk/example_topk.py:34–48` | Repeated row max plus max index reduction; `cross_lane_reduction@1` supplies each reduction | Source removes **all** equal maxima and uses finite sentinel -10000. This is a distinct small-menu method, conditionally top-k only under a suitable unique-value/range binding; it does not establish general tie-preserving top-k. Record that boundary without repairing the example. |
| `examples/norm/rms_norm.py:7–62`; `norm/layernorm.py:7–130` | Full-row retained input versus first-pass statistics and second-pass reload; reduction partition, storage and fused normalization reuse | RMS reload reverses tile order for a cache-locality hypothesis. LayerNorm uses E[x²]-E[x]² with saved mean/rstd, not Welford. Backward owns dX rows and combines dGamma/dBeta partials with initialized atomics. Cancellation/epsilon/statistic precision remain numerical bindings. |
| `examples/online_softmax/online_softmax.py:8–45` | First pass computes row log-normalizer, second pass reloads logits and emits probabilities | Reuse compact row statistics and `softmax_probability_recomputation@1` for the second pass. LSE merge algebra is rescalable; its source reference can be the current tile max, so it is not automatically version 2's running-max policy. Tail-neutral values and exponent range need their own binding. |
| `examples/gemv/example_gemv.py:119–209` | Output-row × reduction-thread decomposition, 128-bit vector chunk per thread, shared atomic combine or explicit thread allreduce | `reduction_partition@1`, `vectorized_access@1`, `cross_lane_reduction@1` explain ownership. Shared atomic initialization/readback require synchronization; all writers need completed reduction. A shared atomic alone does not supply that ordering. |
| `examples/hadamard_transform/example_hadamard.py:37–147` | Butterfly structure plus `staged_relayout@1`; conditional relation to `butterfly_linear_transform@1` | Power-of-two lengths 2–32768; stages inside thread, then warp shuffle, then shared exchange to regroup cross-warp partners. Shared exchange is chunked above a 32 KiB budget. The old rule is normalized and bounded to H≤512; this example is unnormalized and extends to 32768, so normalization/domain need a variant. Compiler-inserted synchronization remains a binding obligation. |
| `examples/convolution/example_convolution.py:20–50` | `implicit_patch_contraction@1`: NHWC output flattening, filter K flattening, `T.im2col` into shared, tiled GEMM | Output sizes depend on stride/dilation/padding; source uses square KH=KW. Match zero-padding and patch coordinates before applying storage/instruction choices. Autotuned sibling changes menus, not this computation organization. |
| `examples/cast/example_per_token_cast_to_fp8.py:9–41`, grouped sibling | Group absmax quantization, granularity 128, E4M3 range ±448, FP32 scale, floor 1e-4; segmented padded storage reuse | Same value/scale pair must reach the consumer. Invalid grouped rows are masked; zero scale and intermediate padding calculations do not authorize consuming padded values. Group offsets and TMA-aligned scale storage are address/layout bindings. Existing `groupwise_low_float_quantization@1` admits only NVFP4/16 or MXFP8/32 with encoded metadata, so this 128-value/FP32-scale pair is a format-domain gap. |
| `examples/elementwise/example_elementwise_add.py:12–32`, `rand/rand_uint.py:9–25` | Pointwise tiling/copy reuse; RNG binds seed, subsequence and per-thread offset before producing 128 values | RNG is an operator with stream identity, not an interchangeable source of arbitrary values. Retain stream/launch-mapping equivalence as a pool gap; no new RNG rule is proposed. |

## Follow a frontend choice to its CUDA owner

`tilelang/language/gemm_op.py:160–192` distinguishes synchronous `T.gemm` from
manual asynchronous WGMMA/TCGEN05 calls. `src/op/gemm.cc:180–268` delegates
instruction/layout/lowering to target implementations;
`tilelang/tileop/gemm/__init__.py:101–154` resolves the same instruction key into
Python layout and lowering classes. CUDA selection is concrete in
`src/cuda/op/gemm.cc:80–97,391–438`: TCGEN05 needs SM100 plus shared/TMEM operand
sites and legal shape metadata; Hopper WGMMA needs compatible types, B shared,
M≥64 and a multiple of four warps. Explicit asynchronous requests fail their
gate rather than silently changing protocol. Two-CTA B shapes cannot fall back
to a one-CTA instruction. This supplies bindings for
`compute_instruction_selection@1`; no generated machine-code capability is
claimed by merely taking a source branch.

`src/cuda/op/copy.cc:778–832` dispatches normal, cp.async, matrix load/store,
bulk 1D/descriptor copy, gather4/scatter4, cluster and TMEM paths through copy
analysis/layout eligibility. Packed FP4 TMA transfer bytes use four payload bits
even when shared slots occupy eight bits (`:51–95`). Thus source shape bytes,
allocated shared bytes and barrier expected transaction bytes can differ.
Reuse transfer/layout/byte-completion rules with the correct units; the complete
native layout menu is an explicit realization gap, not another algebraic rule.

`src/transform/layout_inference/layout_inference.cc:1326–1505` tries free roots
within connected components and scores complete legal assignments. The default
register-count policy and optional IO-aware policy are stated in
`layout_cost_model.h:1–22`. These are compiler search policies, not optimizer
variables freely selected independently for each access. Annotated layouts,
MMA fragments and copy compatibility constrain candidate assignments.

The `examples/aws/` name denotes automatic warp specialization, not a separate
hardware backend. `aws/gemm.py:38–64,119–133` contrasts compiler-generated roles
with a typed manual schedule (TMA warp 0, MMA warp 1, epilogue warps 4–7).
Input K stages and persistent output TMEM stages are independently annotated.
`src/cuda/transform/auto_schedule/role_based_scheduler.cc:5–35` classifies roles,
backward-slices their private dependencies, discovers cross-role ownership
handoffs and emits full/empty pipelines for materialization. Unsupported branch,
fan-out and delayed-WGMMA cases are named source boundaries. Reuse warp roles,
rings and handshakes; inferred dependency placement remains compiler evidence.

`aws/flash_attention.py:92–145` is another concrete
`online_rescaled_reduction@2` binding: lag threshold 8 in base2 units, gain 1,
skip old-output read/modify/write when every covered correction is one, then
TMEM P×V. The skip predicate's collective must cover every affected output row.

`examples/dynamic_shape/example_dynamic.py:20–49` keeps M/N/K runtime while
tiles/transposes/types/stages remain specialization choices. Eager-JIT notebooks
change invocation/binding syntax. Layout visualization, analyzer, autodd and IKET
examples inspect or instrument existing kernels; they add no new operator
refinement. Backend reducer planning/epochs and scan lowering are retained gaps:
`src/op/scan.cc:74–104` requires linear shared scan layouts, and the CUDA
registration delegates cumsum/cummax to common lowering. A generic sum Template
does not encode every prefix result or reducer lifecycle.

All NVIDIA example directories are assigned a design-group disposition in the
[inventory](inventory.md). AMD/Ascend are excluded; external submodule content,
native instruction/layout validation and end-to-end replay remain outside this
source pass. No per-entry audit or benchmark is required for those dispositions.
