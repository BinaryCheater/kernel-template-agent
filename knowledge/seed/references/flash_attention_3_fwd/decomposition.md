# FlashAttention-3 forward (sm_90) — decomposition

Source: Dao-AILab/flash-attention @ `a3a9dfa9bf10b52e55e8507b16ab0fe0dc1f38a0`, directory
`hopper/`. Program: `FlashAttnFwdSm90` (`hopper/flash_fwd_kernel_sm90.h`) with
`CollectiveMainloopFwdSm90` (`hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`) and `CollectiveEpilogueFwd`
(`hopper/epilogue_fwd.hpp`). Configuration: f16, head_dim 128, Arch 90; `tile_size_fwd_sm90`
gives kBlockM = 128, kBlockN = 176 (non-causal) or 128 (causal), MmaPV_is_RS, IntraWGOverlap
(`hopper/tile_size.h:31-33`); kStages = 2 (`hopper/flash_fwd_launch_template.h:47`); two consumer
warpgroups (kBlockM / 64).

**Analysis-only.** wgmma, setmaxnreg and the sm_90a cluster/TMA multicast path do not exist on
GB10 (sm_121), so `runnable_on: []`.

Pool: seed + proposals of the GEMM and FlashAttention-2 references.

## 1. Workload

Identical to `flash_attention_2_fwd` (same Problem name): `O = softmax(scale Q K^T + mask) V` and
LSE, f16, head_dim 128, GQA, optional causal mask.

## 2. Transaction inventory

| # | Class | Transaction | Where | Explained by |
| --- | --- | --- | --- | --- |
| T01 | schedule | persistent grid; static or dynamic (atomic) tile fetch; LPT order and L2 head sections for causal | `hopper/tile_scheduler.hpp:141-215, 220-366`, `hopper/flash_fwd_launch_template.h:61-76` | sched |
| T02 | schedule | tile index fetched by the producer, broadcast via smem with two named barriers | `hopper/tile_scheduler.hpp:327-363` | sched (broadcast) |
| T03 | executor | producer warpgroup (one warp issues TMA) vs two consumer warpgroups | `hopper/flash_fwd_kernel_sm90.h:75-78, 208, 308-446` | ws |
| T04 | storage | setmaxnreg 24 / 240 | `hopper/flash_fwd_kernel_sm90.h:82-83, 309, 361` | regs |
| T05 | executor | each consumer warpgroup owns 64 rows | `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:89-96` | out_wg |
| T06 | compute | QK^T wgmma SS; PV wgmma RS | `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:90-110, 1141, 1177, 1182` | mma_qk, mma_pv |
| T07 | compute | online softmax, exp2 with folded scale, quad allreduce, finalize with __logf | `hopper/softmax.h:36-60, 67-90, 97-160` | online, exp2, row_xlane |
| T08 | numeric / transfer | P -> f16, used as register A operand of PV | `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:1158-1159, 1202` | p_convert, p_reuse |
| T09 | schedule | intra-warpgroup overlap: QK(n), PV(n+1), softmax(n) while PV in flight | `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:1138-1250` | intra_overlap |
| T10 | schedule | inter-warpgroup ping-pong via WarpSchedulerWG named barriers | `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:915-950, 1176, 1183` | pingpong |
| T11 | reduce | n_block range from mask; masked / unmasked iteration classes | `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:608-620, 1206-1236` | causal_prune, kv_part |
| T12 | storage | sQ reloaded per tile after QueryEmpty | `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:812-830, 936-939` | q_smem, kv_handshake |
| T13 | storage | sK, sV with 2 stages; smem_o aliases smem_v | `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:131-141`, `hopper/flash_fwd_kernel_sm90.h:90-101` | kv_smem, kv_stages |
| T14 | transfer | TMA loads of Q, K, V from one thread | `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:749-768, 812-824` | kv_tma |
| T15 | sync | full/empty mbarrier per slot (PipelineTmaAsync) | `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:283-289, 750-768, 1140-1143, 1185-1188` | kv_handshake |
| T16 | transfer | 2-CTA cluster, K/V multicast (non-causal, even tiles) | `hopper/flash_fwd_launch_template.h:210-219`, `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:740-768` | multicast |
| T17 | transfer | EVICT_LAST on K/V, EVICT_FIRST on Q | `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:753, 767, 820` | cache_hint |
| T18 | access | GMMA 128-byte swizzled smem layouts | `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:127-150` | swizzle |
| T19 | transfer + sync | O -> f16 -> stmatrix to smem_o -> fence_view_async_shared -> named barrier -> TMA store | `hopper/epilogue_fwd.hpp:241-265, 312-324` | epi_stage (partial) |
| T20 | transfer | TMA OOB zero-fill on load, clipping on store; LSE predicated | `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:1150`, `hopper/epilogue_fwd.hpp:298-330` | kv_guard, o_guard |
| T21 | schedule | tile size, scheduler class, cluster size chosen per workload flags | `hopper/tile_size.h:31-33`, `hopper/flash_fwd_launch_template.h:41-76, 210-219` | none (ux-10) |
| T22 | access | static assignment of the 16 named barrier IDs | `hopper/named_barrier.hpp:50-59` | none (ux-11) |
| T23 | other | register allocation inside the 240-register budget | `hopper/flash_fwd_kernel_sm90.h:83` | none (ux-12) |

Excluded variants: FP8 V transpose (`Transpose_V`, `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:702-791`),
paged KV without TMA, varlen, split, PackGQA, AppendKV, Qv, local, softcap, RescaleOBeforeGemm
(head dim > 128 only).

## 3. Refinement chain

| Step | Template | | Step | Template | |
| --- | --- | --- | --- | --- | --- |
| online | online_rescaled_reduction | M | acc_store | physical_storage | M |
| fuse | operator_fusion | M | q_smem | physical_storage | M |
| out_cta | output_tiling | M | kv_smem | physical_storage | M |
| sched | persistent_tile_scheduler | N | kv_tma | async_transfer (controller) | M |
| kv_part | reduction_partition | M | kv_stages | multi_buffer_rotation | M |
| causal_prune | masked_tile_pruning | M | kv_handshake | per_slot_handshake | N |
| ws | warp_specialization | N | multicast | cluster_multicast | N |
| regs | register_rebalancing | N | cache_hint | cache_eviction_hint | N |
| out_wg | output_tiling | M | swizzle | shared_resource_binding | M |
| mma_qk | compute_instruction_selection | M | intra_overlap | cross_iteration_software_pipelining | N |
| mma_pv | compute_instruction_selection | M | pingpong | pingpong_schedule | N |
| p_convert | operand_precision_conversion | M | epi_stage | staged_relayout | P |
| p_reuse | accumulator_operand_reuse | M | kv_guard | boundary_predication | M |
| exp2 | approximate_math_rewrite | M | o_guard | boundary_predication | M |
| row_xlane | cross_lane_reduction | M | | | |

Match rate 20/29. 12 unmatched transactions (8 new-Template steps, 1 partial, 3 unexplained).

## 4. Least certain judgements

1. **ws as one step.** Warp specialization changes the issuer of every load and every MMA and
   introduces the role handshake. The skill asks to split such steps; it was kept as one
   Template because no sub-step is meaningful alone (a producer role without the handshake is
   not a program). This is evidence for "a Template may refine several transactions at once".
2. **kv_stages matched with beta = 1.** The consumer release was read as the trailing barrier,
   which makes P = 1 <= S - 2 + beta = 1 hold. The per-slot protocol itself is a separate step
   (kv_handshake). An alternative reading would make kv_stages partial.
3. **cache_hint bound once.** K/V get EVICT_LAST and Q gets EVICT_FIRST; the chain binds the
   K/V policy only. Two chain items would be more faithful; one was kept because the Q hint is a
   single load per tile.
