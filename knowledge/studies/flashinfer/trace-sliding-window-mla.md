# Sliding-window MLA: frozen tactics and row-owned cluster merge

Public `flashinfer.mla.batch_decode_with_kv_cache_mla_swa` reaches the retained
`include/flashinfer/attention/mla_swa_sm90/` tree. All following paths are
relative to that directory in the [pinned source](inputs/sources.json).
README contract: SM90 cluster support, opt-in shared capacity at least 223584
bytes, Page64 BF16 latent+rope rows of width 1088, 64 heads, latent output
1024, window_left=512 (513 inclusive tokens), B in {1,2,4,8,16}, sq in {1,4}.
These are source qualification limits; no tactic was measured here.

## Shared operator and derived menus

`device/softmax.cuh:4-51` maps packed row back to query token/head, sets
q_pos=L-sq+token, and admits tokens max(q_pos-window_size+1,0)..q_pos.
It sums FP32 exponent mass before BF16 P conversion, with defined zero
probabilities for masked/empty support. Existing grouped_head_row_packing@1,
boundary_predication@1 and online_rescaled_reduction@2 gain=1 bind these choices;
empty_support_normalization@1 needs the explicit source empty-state convention.
This is not classic MLA's rounded-P denominator.

`config.cuh` defines six supported complete configurations, not a freely
combinable product of M, S, role count, lookahead and reduction backend:

| Tactic | Mainloop and value ownership | Partial/reduction binding |
|---|---|---|
| M8S9 / M8S5 | Register softmax; QK group owns value slice 0, three PV groups own slices 1..5,6..10,11..15 | FP32 normalized O plus base2 LSE in global memory; separate combine kernel |
| M8S3 | Same score/value organization | FP32 raw numerator/max/mass, row-packed DSM; three final row owners |
| M16S2 | Helper softmax, four lookahead slices, four PV groups interleaving 16 value slices | Raw states in natural-fragment DSM mailboxes over retired Q storage, cluster joins |
| M32S2 | All-group softmax, three PV groups cover contiguous 5/5/6 slices | Raw natural-fragment DSM over retired Q, per-group ready/ack handshakes |
| M32S1 | Same all-group/contiguous organization | Direct output, no cross-split combine |

`config.cuh:18-47` defines two source QK issue sequences: each covers all 17
64-feature latent/rope slices exactly once, with exactly one initial zeroing
operation. `device/qk_sequence.cuh:1-22` waits their WGMMA completion before
score consumption. This binds reduction_partition@1 and instruction/pipeline
realizations, not a new softmax identity. M16's lookahead and helper placement
remain a native multi-frontier scheduling binding rather than a measured free
lookahead domain. `device/softmax.cuh:54-83` and `native_softmax.cuh` place
row statistics/P writers in different roles, preserving same query identity
and full score before exponentiation.

`config.cuh:120-145` derives value-slice ownership. Four interleaved owners of
16 slices differ from contiguous assignment, but output_tiling@2's exact-cover
obligation is unchanged. A QK role that also computes PV retains the mixed-role
graph gap already named in the [Hopper trace](trace-mla-hopper-roles.md).

## Three peers, uneven final row ownership

[row_owned_cluster_state_combine@1](delta/row_owned_cluster_state_combine@1/template.yaml)
and its [proof](delta/row_owned_cluster_state_combine@1/proof.yaml) capture
M8S3's new combine object.

| Source choice | Candidate field / reusable binding | Evidence and boundary |
|---|---|---|
| `config.cuh` RowOwnership; `device/mailbox.cuh:12-18` | Unique final row ownership | Rank r owns [floor(8*r/3),floor(8*(r+1)/3)): 2/3/3 rows. Ownership is not inferred by rounding rows-per-peer to one integer. |
| `device/mailbox.cuh:19-47` publishes only nonowned numerator rows and max/mass | `pre_mapping`, `pre_lifetime` | Own-row numerators remain in registers; other rows are row-packed in dead KV arena. First cluster join publishes all payloads and keeps peers alive. |
| `device/accumulator.cuh:22-33`, `mailbox.cuh:49-74` | Raw-state common-frame invariant | a_s=exp2(m_s-M) for positive mass, otherwise zero; D=sum(a_s*d_s); multiply raw numerator by a_s/D. Multiplying by d_s again is wrong. All-empty output uses zero factors. |
| `device/memory.cuh:110-115` | Row-packed mailbox map | offset=slice*512+row*64+(warp*8+lane/4)*2, then two FP32 words. A source fragment-to-feature realization is still needed beyond this address-map witness. |
| `device/mailbox.cuh:75-132` | Local plus two remote contributions, exact final writer, reader retirement | Owner reads each peer's same row/feature, reconstructs adjacent features with lane exchange and predicates visible output rows. Terminal cluster join protects mailbox reuse and peer exit. |

The [logical witness](checks/check_cluster_row_combine.json) checks all 8192
row-packed words once, all 8192 final row/features once, two remote plus one
local contribution and raw-versus-normalized weighting. Empty states avoid
undefined reference subtraction. It does not execute DSM instructions,
fragment/lane permutations or cluster barriers. smem_partial_combine@1 is
intra-CTA with one-warp/all-thread result placement; split_combine_kernel@1
uses another kernel. Neither supplies this row-owned resident cluster object.

## Related combine and storage choices

`launch.cuh:57-80` uses cluster size S for DSM; global tactics allocate FP32
partials and launch `device/combine.cuh`. That combine weights normalized O
by exp2(LSE_s-maxLSE) and renormalizes. This is concrete split_combine_kernel@1
reuse, with matching base2 LSE and completed first kernel. It is not the raw
numerator/max/mass payload above, even though both compute the same logical
attention result.

`device/mailbox.cuh:147-184` gives M32's separate peer max/mass publication
and acknowledgment. M16 starts at :187 with natural-fragment payload and a
capacity assertion against retired Q. These relatives share the raw-state
algebra but require different mailbox maps/ready-ack protocols; v1 deliberately
does not claim their native layout. physical_storage@1 covers the dead-Q/dead-KV
alias lifetime, and per_slot_handshake@1 supplies the ready/ack obligation once
its actual peers and arrivals are bound. Slice lookahead, per-WG helper timing
and DSM address/barrier semantics remain realization gaps. No need to create
six duplicate merge rules for the tactic menu.

This subgroup now has source-backed computation, tactic and reduction
dispositions, including named native gaps. Continue other MLA cache/target
families; this subgroup is not the full FlashInfer library.
