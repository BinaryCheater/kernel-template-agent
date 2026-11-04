# First trace: grouped row ownership and scaled partials

Pinned source: [manifest](inputs/sources.json). All source locations below refer to that revision. Analysis target: NVIDIA SM90 CUDA, E4M3 FP8 A/B, FP32 raw/final accumulators, BF16 D, K-major operands. No measured GPU facts or selected heuristic winner are claimed.

## Caller and concrete path

`m_grouped_fp8_fp4_gemm_nt_masked` (`csrc/apis/gemm.hpp:266–318`) accepts A `[G, M, K]`, B `[G, N, K]`, D `[G, M, N]`, and an int32 valid-length vector. The host verifies tensor shapes/types and positive expected_m/capacity; it does not read the device lengths to prove that each is within capacity. That remains a caller premise. The SM90 + FP32-SFA branch selects `sm90_m_grouped_fp8_gemm_masked_1d2d`.

Witness: G=3, capacity M=128, N=192, K=256, lengths `[65,0,1]`, expected_m=65, analysis grid=132. A/SFA and B/SFB include per-group capacity. Bind a source-menu configuration BLOCK_M=64, BLOCK_N=96, BLOCK_K=128, cluster=1. The SM90 menu admits these tile extents; the stage count depends on the resource formula. This is a chosen branch for substitution, not a claim that the heuristic selects it for this input.

`csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp` constructs GemmType::MGroupedMasked, builds tensor maps with capacity M and G, instantiates `sm90_fp8_gemm_1d2d_impl`, and launches grid=`num_sms`. `expected_m` feeds configuration scoring; `masked_m` feeds actual tile ownership. These inputs have different roles.

The scheduler's persistent index is `iteration*num_sms+blockIdx.x`. Each group contributes `ceil(valid_m/BLOCK_M)*ceil(N/BLOCK_N)` tasks. In the witness this is 4+0+2=6 tasks. The zero-row group contributes none. Static striding gives each task exactly one CTA iteration. Swizzling orders the two M and two N blocks of group 0 without changing ownership.

For group 0 the second M tile starts at row 64: it contains one required output row and 63 capacity rows. `is_computation_valid` checks the math warpgroup's starting row, not every output row. The TMA D descriptor is bounded by capacity M=128, not length=65, and the device store emits the full tile (`impls/sm90_fp8_gemm_1d2d.cuh:424–439`). Thus group 0 can write rows 0..127; group 2 can write 0..63. Required results cover rows `<masked_m[g]`; capacity tails have no preservation guarantee. A downstream consumer must retain the same validity metadata.

The contiguous API (`csrc/apis/gemm.hpp:180–251`) instead accepts A `[M, K]`, B `[G, N, K]`, D `[M, N]`. Without PSUM, its row-ID vector has length M and must respect tile grouping; one B group is selected at the tile origin. With PSUM the vector holds G end offsets; a group starts at the aligned previous end, not at the previous end itself. This needs separate source-backed alignment and padding premises before a complete grouped Template is proposed.

## Decisions and bindings

| Source location and observed choice | Rule / field or gap | Evidence and remaining boundary |
|---|---|---|
| `csrc/apis/gemm.hpp:203–218,287–318`: shape/type contracts, arch9 + FP32 SFA | Workload boundary and capability gate | FP4 aliases do not establish SM90 FP4 support; actual type validators and branch must admit the operands. |
| `csrc/apis/gemm.hpp:294–295`: expected_m positive; masked vector remains device data | Dispatch heuristic vs runtime metadata | Expected size can change chosen configuration; it does not replace actual group lengths. Validate length bounds through caller evidence/GPU boundary checks. |
| `scheduler/gemm.cuh:193–217`: prefix of ceil group tile counts; `++current_iter` grid stride | `persistent_tile_scheduler@1.fetch=static_stride`, `grid=132`; grouped decoder remains a gap | `grouped_ownership.py`: 864 masked domains, including empty groups. No planner descriptor array is produced. |
| `scheduler/gemm.cuh:112–150`: primary-axis groups, width8/16 and odd SM90 multicast fix | `block_rasterization@1` provisional comparison; exact decoder binding open | 240 swizzle domains recover all tile coordinates; width is a compile-time menu choice selected by estimated operand footprint, not a measured optimum. |
| `scheduler/gemm.cuh:153–184,324–338`: group/global addresses and computation-valid predicate | Workload ownership; `boundary_predication@1` partial match | Required rows and permitted tile-store rows differ. Treating length as exact store extent fails the witness. |
| `scheduler/gemm.cuh:218–267`: M-prefix alignment; physical K start vs SF-K prefix | Grouped address contract gap | M ends `[65,193,321]` give starts `[0,128,256]`; K lengths `[1000,4096,900]` with align128/gran128 yield physical starts `[0,1024,5120]`, packed-SF starts `[0,2,10]`. Last is the SM100 span512 specialization, separate from the SM90 span128 default. |
| `heuristics/sm90.hpp:16–114`: M/N menus, multicast filters, minimum stage filters | Compiled menu; legal guards vs performance filters | BLOCK_M64/N96/K128 is a source-admitted witness. Stage minimum3/4 is a policy for latency hiding; the shared-memory capacity bound is a resource constraint. Winner replay remains open. |
| `heuristics/sm90.hpp:154–201`: stage count from capacity and extra/per-stage bytes | `physical_storage@1`, `multi_buffer_rotation@2.S` after domain check | Stage count is derived by this implementation from layout and storage, not independently selected. No copied bandwidth estimate is a KTA measurement. |
| `impls/sm90_fp8_gemm_1d2d.cuh:126–205`: full/empty barriers, stage ring, producer TMA | `per_slot_handshake@1`, `async_transfer@1`, `multi_buffer_rotation@2`; binding pending | Producer waits empty phase^1; consumer waits full phase. Empty arrivals include all participating warps/CTAs. The barriers must still progress when computation is skipped. |
| Same device body `:139–143,169,216–218`: TMA/math register budgets | `warp_specialization@1`, `register_rebalancing@1` | Source fixes producer40, math248/232 by math thread count. No free register-budget family is inferred. |
| Same device body `:228–247`: a BLOCK_N tile may cross B's per128 scale boundary | New `block_scaled_partial_accumulation@1.gran_n=128`, metadata owner | At N block1/tile96, columns96..127 use B scale0,128..191 use B scale1. `num_former_iters=4`, `num_full_iters=12` in groups of 8 columns. |
| Same device body `:285–344`: raw dot reset then scale promotion each K128 | New rule `partial_k=gran_k_a=gran_k_b=128`, `gran_m=1`, `gran_n=128` | Independent rational decode vs grouped promotion agrees for two unequal K-scale groups and boundary columns. Reset is passed as `k=0` to the first MMA of each partial. |
| `mma/sm90.cuh:14–32`: k0 means ScaleOut::Zero, later K32 instructions mean One | `compute_instruction_selection@1`, source Realisation boundary | Raw128 partial consists of four K32 MMAs; The selected CUTLASS body is now retained; see trace-scale-storage.md. No emitted PTX/SASS claimed. |
| Device `:351–439`: skipped computation drains barriers, BF16 conversion, shared swizzle, fenced TMA store | `staged_relayout@1`, `boundary_predication@1`; numeric conversion and exact fragment map pending | Store waits protect previous use of smem_d. Lane/fragment mapping needs coordinate replay against the retained CUTLASS body. |
| `csrc/apis/layout.hpp:35–56`, `impls/smxx_layout.cuh:87–196`: FP32 scale broadcast/transpose/exponent pack | Existing prepack/bitstream rules only partial matches | Humming bitstream rule requires groups of32 codes and does not cover a short group of four scale bytes. A distinct parameterization or revised version needs evidence; no silent reuse. |
| `impls/smxx_layout.cuh:140–170` vs `docs/scaling-factor-format.md` §7.3 | Open discrepancy: gap-row input reads | Pack selection substitutes zero for gaps, but preceding global-load loops copy the tile's SF rows into shared memory before testing valid_mn. Documentation's no-input-read wording exceeds this source path. Requires a real uninitialized-read/sanitizer check; no memory-safety claim. |
| `impls/sm100_fp8_fp4_gemm_1d1d.cuh:51–97,196,409–439`: MXF4 vs mixed FP8/FP4, K32/128 scale selectors | Separate instruction Realisation; not the SM90 post-partial algorithm | Source uses scaled UMMA and independent kGranKA/B. K-grouped requires matching grains. Tensor-memory scale and accumulator lifetimes need their own full trace. |

## Composition and checks

The partial numerical chain is in [delta/chain.yaml](delta/chain.yaml): K partition → scale-owner promotion. Producer/consumer edge: each raw partial supplies the same output coordinate's final accumulator. Metadata edges bind (group, m, n, K interval) to its A/B scale; the scheduler must not substitute a numerical group for a routing group. Barrier/order edges protect raw operand storage and the shared epilogue buffer; they are not yet represented by a complete Chain.

[scaled partial witnesses](checks/scaled-partials-result.json) reject final-only scaling, stale raw accumulators and wrong B scale ownership. [Grouped witnesses](checks/grouped-ownership-result.json) establish bounded source-index arithmetic. No CUDA kernel was compiled or run. Full-library inventory rows remain open and admission remains independent.

Continued source tracing: [scale storage and SM100](trace-scale-storage.md), [einsum views and combined reductions](trace-einsum.md). The later trace resolves the previously missing direct submodule source boundary; compilation and device behavior remain open.

Attention continuation: [dense/paged/sparse MQA](trace-attention.md), including scale/ReLU ordering, union incidence, compressed output and paged cache bytes.

Hyperconnection continuation: [prenorm and Mega MHC](trace-mhc.md), including shared BF16 boundaries, epoch completion and finite coefficient rescaling.

MegaGate continuation: [score production and routing](trace-gate.md), including sorted-run top-k, bias/weight separation and random/fixed routing boundaries.

MegaMoE continuation: [buffers, dispatch and fragment readiness](trace-moe.md). The full family remains in progress.

Locality continuation: [allocation/probe mapping and queue counterexample](trace-locality.md).

Legacy/auxiliary continuation: [Triton families, TileLang baseline and IDE target](trace-legacy-auxiliary.md).

cuBLASLt continuation: [adapters and external implementation boundary](trace-cublas.md).

Native/epilogue continuation: [BF16, per-column FP8 and shared output operators](trace-native-epilogues.md).
