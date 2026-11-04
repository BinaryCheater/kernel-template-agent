# Hopper MLA: mixed roles and shared decoded staging

Route: FA3 MLA backend -> generated SM90 module -> `csrc/batch_mla_sm90_run.cu`
-> `include/flashinfer/attention/mla_hopper.cuh` in the retained
[source](inputs/sources.json). Line references below are within that header.
Representative: query tile 64, key tile 64, two stages, latent 512, rope 64,
two 128-thread groups and BF16 compute; E4M3 KV optionally decodes to BF16.

| Source decision | Existing field / gap | Concrete binding and disposition |
|---|---|---|
| :68-174,1280-1305: two raw KV slots, rope/P union per slot, one decoded staging region across all slots, final O overlay outside the ring | physical_storage@1; decoded_tile_materialization@1 `pre_lifetime`, `pre_binding` | FP8 predicate is the exact E4M3 type, not sizeof(KV)==1. Decoded latent=512 and rope in {0,64}; arbitrary byte codes do not inherit conversion support. Output staging aliases the whole retired data path, not one separately reserved output buffer per raw slot. |
| :153-170: raw FP8 rope has four 16-byte blocks per 64-element row, versus eight for BF16; use 64-byte rather than 128-byte raw swizzle | Layout/instruction realization obligation | Dtype changes row bytes, active lanes and load count together. Source names alone do not prove injectivity; both raw writes and decoded reads must bind the new swizzle. |
| :867-880: Q channel producer arrivals 128/consumer arrivals 128; KV producer arrivals 128/consumer arrivals 256 | per_slot_handshake@1 `full_signal=cp_async_arrive`, `empty_arrivals`, `stages=2`; multi_buffer_rotation@2 | Q has one consumer group; KV has two because both groups perform PV. Shared trait ancestry does not preserve the classic MLA role graph. Exact copy/barrier arrival semantics remain a backend obligation. |
| :894-1048: group 0 loads Q/KV and computes one value-feature half; :1055-1257 group 1 computes full QK, softmax and the other PV half | Mixed producer/compute role gap | warp_specialization@1's program and summary assign exclusive load versus compute roles; those fields cannot express a producer that also owns output arithmetic. score_replication_or_exchange@1 supplies replicated or key-sharded QK, not one complete score producer feeding both PV owners. This is a separate single-score-producer/mixed-role binding gap. |
| :1092-1145 and analogous loops: group 1 waits raw KV, decodes shared BF16 staging, completes QK, writes P over retired rope, publishes output correction | decoded_tile_materialization@1; physical_storage@1; online_rescaled_reduction@2 with gain 1 | Two CTA joins surround decode. Group 0 passes through them. QK completes before rope/P alias overwrite. P and correction belong to the same row/frame generation. |
| :990-1040,1127-1149: both groups correct old O, wait/read common P+latent, complete PV and release KV | per_slot_handshake@1 arrival/lifetime obligations | Completion of both PV groups, not QK alone, releases latent reuse. FP8 additionally joins both groups after PV, before the single shared decoded staging can hold the next raw slot. Increasing raw ring stages does not multiply decoded staging capacity. |
| :594-620 and :1121-1127: mass sums FP32 exponent weights before converting P | online_rescaled_reduction@2 numeric binding | This differs from classic MLA rounded-P mass in score_replication_or_exchange@1. Do not transfer that numeric premise merely because both return the same attention interface. No numerator-only gain is present here. |
| :1029-1048,1226-1257: group 1 publishes final m/d; group 0 waits kMDReady; each writes its output half | output_tiling@2 plus common-state publication | Both output halves normalize with one completed row state. Final O overlay needs all data readers retired, and both groups finish stores before the next descriptor. |
| :911-931,1077-1091,1226-1257: no KV tiles leaves KV channel unchanged, but Q channel and final state/output protocol still execute | empty_work_pipeline_projection@1 mixed channel projection | Both roles derive the same key-tile bound. No P-ready or per-tile OScaleReady is fabricated; Q release, final MD handoff and empty output remain owned. Next nonempty work must retain matching Q/KV phases. |
| :1261-1274: cooperative grid synchronization, then in-kernel persistent split merge | Existing split-state algebra; cooperative merge realization gap | Every role/CTA reaches the join and launch residency/progress must be valid. A separate combine-kernel rule cannot supply these device events. |

This source-backed role/lifetime disposition does not need a new logical test:
no native event mapping or conversion claim is promoted. Existing logical
attention checks cover only their declared arithmetic domains. Remaining
choices needing extraction are the single-score-producer/mixed compute graph
and the cooperative phase transition. Native wait/arrival counts, decoded
swizzle coordinates, instruction completion and empty-to-nonempty reuse remain
implementation evidence obligations. `mla_swa_sm90/`, CuTe DSL/cuTile and
SM120 compressed cache relatives stay separately open.

## Next representative boundaries

`mla_swa_sm90/README.md` declares a separate dense Page64 BF16 sliding-window
contract: latent 1024 plus rope 64, 64 heads, Q positions L-sq+i and an
inclusive 513-token window. Its six frozen tactics couple M, split count,
softmax ownership and global/DSM/direct reduction; they are not a Cartesian
parameter menu. `device/mailbox.cuh:1-80` gives an M8/three-CTA representative:
CTA rank r owns rows [floor(8*r/3),floor(8*(r+1)/3)), publishes nonowned row
partials through DSM mailboxes, joins the cluster, then forms common-frame
weights from every rank's max/mass. Read the rest of that row-owned combine
and its register/helper/all-group softmax relatives next; current observation
is a source-bound starting point, not a completed tactic-family disposition.

`flashinfer/mla/_sparse_mla_sm120/_api.py:16-43` distinguishes model cache
families (DSv3.2, GLM-NSA, DSV4, GLM53_NOPE, DOTS3_SWA and DSV4.1), ordinary
FP8/BF16 versus separately keyed NVFP4 formats, and configuration profile
hints versus the authoritative C++ dispatcher. Probe-layout profiles do not
supply measured costs for other legal strides/index layouts. The next study
must follow actual cache payload/scale layouts and selected decode/prefill
consumer, rather than deriving one universal quantized-MLA rule from this facade.
