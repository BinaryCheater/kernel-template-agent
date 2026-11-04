# GEMM first pass: scope, existing bindings and atom-owned multicast

The [GEMM inventory](gemm-inventory.md) and [source discovery](inputs/gemm-source-scope.json)
expand the full-library pass beyond attention. Retained paths below are relative
to the pinned [tree](inputs/sources.json). No absent CUTLASS dependency is fetched.

## Wider clusters have three different rank maps

Retrieval cue: a physical GEMM cluster contains several independent2-CTA matrix
instruction atoms, while operand tiles or scale factors are shared across atoms.
New [atom_owned_cluster_multicast@1](delta/atom_owned_cluster_multicast@1/template.yaml)
and its [proof](delta/atom_owned_cluster_multicast@1/proof.yaml) separate:

- P: shared-memory payload recipients;
- O: full-barrier owners, projecting each recipient to its adjacent even-rank atom leader;
- R/E: producer empty-barrier targets and the number of consuming-atom arrivals.

Native representative is `flashinfer/prims_ts/batched_gemm/batched_gemm_cga.py:448-583`.
Rank=x+X*y, atom owner=rank&~1. The helper derives these source relations:

| Operand binding | Payload relation | Empty release relation | Expected atom arrivals |
|---|---|---|---|
| A/SFA | One X column across Y | Both CTAs of its producer atom at Y=0 | Y |
| B | Issuer X parity across atoms in one Y row | Both CTAs of the first atom in that row | X/2 |
| SFB | Whole X row | Sole X=0 producer in that row | X/2 |

Full owners always use O=(P|(P>>1))&0x5555. In a4x2 cluster, SFB issuer0
sends data to{0,1,2,3}, arms owners{0,2}, releases target{0} and expects2
atom completions. Odd-parity A issuer1 instead sends to{1,5}, arms{0,4},
and releases{0,1} with2 arrivals. Neither payload nor owner count identifies
all synchronization destinations.

`mixed_cga_ts.py` independently selects task CTAs and atom-relative pipeline
routing. A task-selection mask is not a TMA destination mask.
`batched_gemm_cga.py:587-650` constructs an exact CLC consumer census from
actual tasks/warp counts; fused resource aliases add no extra participants.
Native byte accounting must include every parity/slice issuer feeding an owner,
and one atom completion must wait for all its readers before fanout release.
The logical maps alone do not prove those native instruction events.

Flexible fallback queries active physical dimensions and recomputes every mask
and count. A4x2 SFB path falling back to2x1 changes P/O/E to{0,1}/{0}/1;
retaining the preferred shape's two-arrival count would stall. Combined A+B
release is explicitly outside the one-sided source relations; the source notes
that it would require a union relation and Y+X/2-1 arrivals.

cluster_multicast@1 supplies the general shared-tile load and reader-retirement
idea, but has a size1/2/4 menu and no distinction between physical cluster and
2-CTA instruction atom. Its fields cannot express these three maps, so this is
a separate refinement rather than an unsupported wider binding of that rule.

The [logical check](checks/check_atom_multicast.json) enumerates elected
issuers for2x1/4x1/2x2/4x2/4x4, owner projection, atom-release census and fallback.
Native TMA/UMMA byte/commit semantics and cluster retirement remain obligations.

## Existing scale and ragged-work rules have concrete bindings

Representative:
`flashinfer/gemm/kernels/cutile/ragged_block_scaled_bmm_cutile.py`.

| Source choice | Existing rule and field | Binding / applicability |
|---|---|---|
| :75-111: expert counts from live m_indptr; flattened CTA tile cursor advances by resident grid size | online_group_work_decode@2 search=scalar_forward,cache_shapes=false plus static persistent traversal | With immutable per-launch counts and nonoverflowing prefix arithmetic, group counts=ceil(M_e/BM)*ceil(N/BN); skip zero groups and decode only the matching expert prefix. Grouped-M swizzle changes local tile order, not expert ownership or product coverage. |
| :109-118 and host :374-382 | physical_storage@1 / tiling address premise | A/C use m_start//BM as a tile index. This is exact only when the selected BM divides each segment origin. The caller's segment_alignment promise bounds legal BM; a128-aligned origin cannot use BM256 by truncating it to0. |
| :128-194, defaults :209-245 | block_scaled_partial_accumulation@1 | Bind partial_k=gran_k_a=gran_k_b=128, gran_m=1, gran_n=128 for the default BM/BN/BK=128 case. Reset raw MMA for each K tile, multiply that partial's scales, then add to final FP32 acc. BF16 output cast follows the full K reduction. |
| :384-408 | `c_format`, `scale_owner` obligations | Source verifies B N-scale granularity and optional A K-scale granularity against BK. A single B scale is broadcast over the N tile, so that tile must stay inside one B scale group; custom overrides that cross groups need another map/promotion binding. Do not assume host shape guards establish every override's MN constancy. |
| :141-142 | tensor_axis_permutation@1 | B tile is reshaped from[1,BN,BK] to[BN,BK], then permuted to[BK,BN] for MMA; metadata stays in the original logical scale coordinates. |

The entry has stale prose about max_m_device; the retained loop actually walks
expert prefix counts from m_indptr. This trace follows that body, not the comment.
The tiny alignment/scale witnesses in the logical check distinguish truncating
segment origins and applying only one scale after multiple unequal K partials.
No unrelated host-edge investigation is required for these bindings.

## Grouped/native/provider boundaries

`grouped_gemm_contiguous_blackwell.py:89-95` states aligned expert M regions;
its loader uses gidx_mapping at a selected tile's first row (:1152,1206,1323).
Every selected tile must belong to one expert throughout its extent. FP8 raw
MMA, FP32 scale promotion and separate/fused accumulator epilogue roles are
retained native designs; deep-K grouping and role tradeoffs remain representative
work in the inventory. A familiar grouped-GEMM API does not prove the same
metadata map as ragged m_indptr traversal.

`gemm_base.py:_SwapAB` records weight/token tile axes, target-specific shape
menus and split-K cluster count. tensor_axis_permutation@1 explains operand/view
orientation; reduction_partition@1 explains K shares. The exact generated
cluster combine remains its own representative; dispatch rows are not speedup
evidence. `_SwapAB` does not authorize arbitrary unsupplied dimensions.

`include/flashinfer/gemm/tgv_gemm.cuh:520-565` peeks the next full barrier before
issuing the current MMA, retains that Boolean, and blocks only when the next
stage is actually needed and not ready. Slot/phase advances on ring wrap. This
binds per-slot readiness and latency-hiding intent; native nonblocking PTX and
instruction issue order remain realization obligations.

`grouped_mm/core.py` maps contiguous expert row intervals through m_indptr and
selects cuDNN or explicit Cake backends. cuDNN's implementation remains external;
its interface does not supply native group scheduling. The generic headers
`include/flashinfer/gemm/group_gemv.cuh` and `group_gemm_lora.cuh` contain only
TODO namespaces. They are not connected GEMV/LoRA kernels; implemented low-rank
families are separately listed under SVDQuant/Cake routes.

Next: the retained grouped FP8 promotion/epilogue design, task-scheduled routed
operands, low-bit/SVDQuant families and deterministic grouped training. Keep the
attention/training reconciliation visible while continuing full-library coverage.
