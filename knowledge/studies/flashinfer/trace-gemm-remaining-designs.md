# GEMM reuse and remaining realization boundaries

Use the existing tile, scale and reduction rules when their fields express the
source choice. The remaining GEMM survey distinguishes four concrete bindings
from native layouts/protocols that those versions cannot yet express. Paths
below are relative to the retained [tree](inputs/sources.json); this extends the
[GEMM inventory](gemm-inventory.md), not the scope of a provider dependency.

| Need / source representative | Concrete rule binding | Boundary |
|---|---|---|
| Small-M router scores: `csrc/dsv3_router_gemm.cu:8-88` -> `include/flashinfer/gemm/dsv3_router_gemm.cuh:42-160` | One CTA per expert,128 threads, VPT8 BF16 values per16-byte load; `vectorized_access@1.vector_bytes=16`. Each thread reuses its expert-weight vector across1..16 token accumulators. `cross_lane_reduction@1(lanes=32,result_to=all_lanes,combine_op=add)` then `smem_partial_combine@1(topology=serial,participants=4,combine_op=add,result_to=one_warp)`; only thread0 stores | K divisible1024, aligned packed operands and declared output cast. Source produces raw scores: no softmax, top-k or normalization is fused here. FP32 accumulation/reduction order must be admitted; the rules' associative real-sum premise does not prove bitwise sequential-FP32 equality |
| Alpha/beta GEMM: `gemm/kernels/cutile/gemm_alpha_beta_cutile.py:122-190,407-409` | `operator_fusion@2(consumer_kind=elementwise,intermediate_site=registers,fused_ops=linear_combination,checkpoint=none)`. Apply alpha*FP32acc+beta*FP32(oldC), then one output cast. `EPILOGUE_SUBTILE=1` stores disjoint N/2 halves | `HAS_BETA=0` removes the old-C read entirely, including beta=-0. It is not evaluation of0*NaN. For nonzero beta, oldC must survive until its own tile load; no forbidden alias with live operands. Subtile conversion must preserve coordinates/order |
| Masked persistent batches: `gemm/kernels/cutile/masked_scaled_bmm_cutile.py:125-160,283-291` | Static CTA-stride walk over Q*ceil(min(max(masked_m),M)/BM)*ceil(N/BN) candidate tiles. A group-M swizzle changes traversal; exact per-batch `masked_m` rejects whole inactive M tiles. Physical matrix tails use zero-padding loads | Scheduling bound and per-batch counts must be one immutable generation. A live boundary tile writes all allocated physical rows in that tile: no per-element masked_m store exists. Invalid semantic rows must be unspecified/unobserved or otherwise owned, not promised untouched/zero. This is a tile-level pruning binding, not a `boundary_predication@1` claim for semantic output rows |
| Hardware-scaled masked MMA: same file`:163-180,243-283` | Complete code and scale tile -> matching in-register scale unswizzle -> `mma_scaled` with FP32 accumulator -> final output cast. Decode/scale coordinates remain paired | Mixed FP8xFP4 route explicitly casts A's E4M3 codes to E2M1 before MMA while retaining A scales. It is a new quantization checkpoint relative to direct mixed-format MMA, not a lossless storage conversion. Native rounding/range and this extra error must be admitted |
| CastA mixed operand: `prims_ts/batched_gemm/tmem_sf_resources.py:799-1019` | Packed MXE2M1 A plus UE8M0 scales -> explicit BF16 pair decode and BF16 scale multiply ->128x256 BF16 tile in TMEM -> ordinary BF16 tcgen05 MMA | Related materialize-once choice, but `decoded_tile_materialization@1.decode_site` has only fragment_registers/shared_tile and its storage effect is SMEM. TMEM site/allocation, tcgen05 stores, async-view fence and reader lifetime are an explicit version/realization gap, not an exact existing binding |
| Tiny GEMM: `csrc/tinygemm2.cu:154-336,364-400` | Four compute warps own disjoint K residues; four elected weight issuers and four activation issuers use separate full barriers and common consumed barriers. Intra-CTA FP32 partial sum, optional BF16 bias then BF16 output reuse `smem_partial_combine@1` and local fusion | Physical16 stages split into four local rings: slot=warp_residue+4*local_slot, local stages4. Each channel can use `per_slot_handshake@1(full_signal=tma_tx_count,empty_arrivals=32,stages=4)`, but the shared release means both weight and activation reads finish before either producer reuses the slot. A complete two-channel join/native role binding remains necessary; `warp_specialization@1` cannot express8 producer warps directly |

The masked scale unswizzle has a concrete coordinate map. For
row r=128*rm+32*q+lane and scale-column j=4*rk+t, q,t in0..3 and lane in0..31,
the raw within-block linear coordinate is16*lane+4*q+t. Reshape
`[Rm,Rk,32,4,4]`, permute to`[Rm,4,32,Rk,4]`, then flatten to
`[BM,K_scales]`. B uses the same inverse map and then transposes to
`[K_scales,BN]`. The physical trailing `[2,256]` dimensions hold the same512
coordinates. This is an explicit reshape/permutation, with tile origins and
padding fixed by the chosen source configuration; arbitrary custom tiles do
not inherit legality from the default128-wide configurations.

CastA's scale lookup is also explicit: for a logical row r and scale-word w,
the shared-memory word index is128*w+4*(r%32)+(r%128)//32. Decode uses the matching
byte for each32-value K group. The source retains explicit nibble decode ->
UE8M0-to-BF16 conversion -> BF16 multiplication because a fused native lowering
has a recorded context-dependent compiler failure (`:887-893`). Algebraic
identity of isolated conversion instructions is therefore insufficient evidence
for replacing this implementation. Special scale bytes and extreme BF16 products
retain their native numerical obligations.

Tiny GEMM's PDL dependency is operand-specific. At`:205-209` only activation
issuers wait for the predecessor; weights and bias can be read earlier. This
requires weights/bias to be independent of that predecessor. Its early launch
completion signal permits successor launch, not early observation of unfinished
output. The per-channel readiness join and final block-wide barrier still govern
consumption and output. Dispatch windows in`gemm/routergemm.py` choose reference,
SM100, tcgen05 or wgmma bodies for concrete shapes; those are source policies,
not new mathematical operations or locally measured performance facts.

No new candidate is needed for the router reduction or alpha/beta fusion.
Named gaps remain for TMEM materialization, two-channel native joins, cluster
split-K output ownership and target-specific mixed-scaled-MMA realizations.
The retained CUTLASS/CuBLASLt/cuDNN/precompiled wrappers define provider
boundaries; absent dependency bodies remain absent. Continue with standalone
quantization/layout and the non-GEMM families while retaining these gaps.
