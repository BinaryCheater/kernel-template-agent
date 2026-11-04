# MoE dense GEMM: expert work and immutable transfer metadata

When many persistent workers revisit the same small set of problem shapes, a
shared immutable descriptor table can replace private descriptor rebinding. When
work sizes differ by problem, retain an inverse-prefix cursor, but distinguish an
empty output domain from a nonempty output whose reduction is empty. These are the
retrieval cues for this slice.

Scope: `torch_grouped_mm.py`, shared `moe_persistent_scheduler.py`, dense portions of
`moe_sched_extension.py` and `moe_utils.py` in
`examples/python/CuTeDSL/cute/blackwell/kernel/moe/`. All five files, including the
scaled entry, are pinned in [the inventory](inputs/moe-variants.json). Scaled construction/layout and device bodies are covered by the
[metadata](trace-moe-scaled-metadata.md) and [device](trace-moe-scaled-device.md)
companions. Paths below are
relative to the retained revision in [the source manifest](inputs/sources.json).
This is source extraction; the original files remain unchanged.

## Workload and selected implementation

Target is SM100 tcgen05 MMA, FP16 A/B/C and FP32 accumulation. The witness has four
experts, hidden128/intermediate256, token lengths[8,0,40,16] and cumulative **ends**
[8,8,48,64] with no leading zero. Use MMA128×128×64, cluster1×1×1,
use_2cta_instrs=false, separate_tensormap_init=true, accumulate_on_output=false and
no bias. [The replayed binding](inputs/moe-dense-binding.json) retains both scenario counts. Forward accepts A(64,128), B(4,128,256), C(64,256). Backward accepts
A(128,64), B(64,256), C(4,128,256). Legal strides, offsets and transfer layouts are
required independently of the scalar scheduler.

In forward, M=tokens_i/N=intermediate/K=hidden. In backward,
M=hidden/N=intermediate/K=tokens_i. Some scheduler comments interchange the fixed
axis names; the executable count/decode code and actual caller determine this
binding. At this tile size forward counts are[2,0,2,2], backward counts[2,2,2,2],
and backward K-tile counts[1,0,1,1]. Empty expert1 owns no forward output but two
backward output tiles that must become zero. The source expert table must be
nonempty; its initial cursor reads count0 even when expert_cnt=0.

## Source decisions and their owners

| Decision and source | Binding or derived quantity | Rule field, invariant or gap |
|---|---|---|
| Dense constructor:88–175; caller:328–455 | M64/128 for one CTA or128/256 for two; N32..256 step32; each cluster dimension power of2, product<=16, even clusterM for two-CTA instructions. K×operand_bits divisible by256, then native MMA-K divisibility checked at:208–212. | Compiled source menu and constraints, not measured preferences. Bias is rejected; output accumulation is rejected for forward. Source shape checks are not a complete capability/stride query. |
| Caller:360–427; extension:190–252 | GEMM-domain layouts preserve actual strides. Forward A receives token M offset, B receives expert L offset. Forward C uses an expert descriptor and no repeated token offset. Backward AB use expert descriptors/local K; C receives expert L offset on the global descriptor. | Source coordinate glue checked against identifiable addresses. Encoding a token base in the descriptor and adding it again in the consumer is invalid. Generic tensor-axis permutation does not by itself select this problem/domain ownership. |
| Scheduler:417–462,478–572 | Grid(clusterM,clusterN,max_active_clusters); initial work=blockIdx.z, stride=number of persistent clusters. Retain expert/start/end and advance over counts as queries increase or repeat. | [online_group_work_decode@2](delta/online_group_work_decode@2/template.yaml), search=scalar_forward/cache_shapes=false. Version1's32-lane implementation is preserved at search=warp_prefix_32; [migration](delta/online_group_work_decode@2/migration.md). persistent_tile_scheduler@1 only supplies single-CTA stride intent; cooperative cluster grid/cost remains explicit glue. |
| Scheduler:576–600,647–695 | Forward ceil(tokens/clusterMtile)×ceil(intermediate/clusterNtile); backward fixed ceil(hidden/clusterMtile)×ceil(intermediate/clusterNtile), even for K0. K tiles derive from hidden or tokens respectively. | Family count construction, immutable ends and Int32 bounds. Do not substitute zero-token pruning for the backward output count. segmented_axis_work_decode@1 fixes a physical segmented output axis and cannot supply a reduction-segment interpretation. |
| Scheduler:604–643 | Smaller tile-count axis changes fastest; ties use M fastest. CTA coordinates=cluster coordinate×cluster shape+peer coordinate. | Exact bijection checked for both branch orders. block_rasterization@1 order is compile-bound and its launch-map cost does not supply a runtime per-expert shortest-axis chooser. Keep that chooser/cluster lifting as an explicit traversal gap; no L2 speedup or old cost formula is claimed. |
| Dense caller:531–550,585–658; utilities:394–555 | Separate32-thread setup block per expert; forward tuple C1, backward tuple AB2; workspace128×experts×tuple_width. Consumer main kernel follows on the same stream. | New [precompute_transfer_descriptors@1](delta/precompute_transfer_descriptors@1/template.yaml). Immutable problem-key table, disjoint writers, completion/visibility before use, no old readers during rebuild. `rebind_transfer_descriptors@2` instead owns private mutable worker slots and cached keys; its fields cannot select this shared table. Native proxy visibility and zero-extent construction are open. |
| Dense CLI:1975,1998–2002; kernel:917–1096 | CLI forces separate setup on. No construct_and_write call exists in the main kernel when direct callers disable it. Scheduler prefetches descriptors for later valid work, not the first publication. | Unsupported fused/inline setup is a source gap. Prefetch is a hint, not descriptor construction or publication. runtime_work_descriptors@1 emits per-work ownership metadata, not hardware transfer descriptors. |
| Dense constructor:127–145; pipeline:795–842,909–915 | Seven warps: four epilogue, one MMA, one TMA, one scheduler. Scheduler ring has2×16-byte records with four Int32 fields;32 producer threads and192 consumer threads per CTA. | per_slot_handshake@1 supplies abstract full/empty protection under matching participants. warp_specialization@1's four-warp consumer unit does not express a single MMA warp plus scheduler. Actual role/resource mapping remains a gap. No CLC or cancellation is selected. |
| Dense resource policy:251–314 | Accumulator stages2, initial output stages2; AB stage count from available SMEM after barriers/output/scheduler. Output stages increase using leftover SMEM. TMEM columns and transfer bytes derive from layouts/CTA group. | Derived policy and coupled budgets, not independent unconstrained knobs. multi_buffer_rotation@2 ends at S8; any derived AB stages above8 need another supported scope. Positive counts/layout legality and native occupancy remain required; no scalar footprint estimate is a measurement. |
| Dense TMA/MMA:994–1170 | AB loop count resets per output tile while ring generation persists. Raw MMA uses ACCUMULATE=false on the first K tile. K0 issues no MMA, skips accumulator acquire/commit/advance. | Existing async_transfer@1 global→shared interface and handshake/ring lifetime apply conditionally. Resetting a tile-local counter does not reset physical slot phase. Native multicast completion, fragment maps and progress are backend obligations. |
| Dense epilogue:1291–1345 | K0 backward skips TMEM wait/read and starts acc_vec at zero. Every output tile, including K0, advances the C-subtile clock; accumulator clock advances only for positive K. | Explicit source identity path is checked; no uninitialized-accumulator claim for backward. [staged_async_output_store@1](delta/staged_async_output_store@1/template.yaml) owns source-ring lifetime for the overwrite path; expert descriptor clipping is an implementation premise. |
| Dense constructor:105; host:1715–1720 | fixed_expert_cnt is stored but not otherwise consumed by this kernel. The tester uses static_E to select whether offs/workspace layouts are dynamic. | Layout specialization control only; no source evidence of a different expert algorithm or a measured specialization benefit. |
| Dense output:493–502,1302–1361 | Optional backward accumulation chooses CopyReduceBulkTensorTileS2GOp; convert partial to C dtype before update. Host zeros C for its accumulation test at:1629–1630. | Retain reduction-store arithmetic as a distinct family variant. Humming first_writer_atomic_combine@1 overwrites with a designated first partial and requires>=2 partials; it does not preserve an arbitrary existing C here. Native update semantics/rounding and repeated-invocation intent need a supported implementation binding. |
| Dense scheduler publication:926–970; pipeline `sm90.py:338–356` | Initial result published unconditionally, then valid results, then an unconditional sentinel. No producer-state advance follows the final commit before producer_tail. | Source protocol gap: tail requires the next useful state, but receives the last published state. An initially invalid worker also publishes two invalid records although consumers exit after the first. Retain counterexamples; do not infer correct progress from the scheduler's index coverage. |

## Composition and discriminating checks

The [partial chain](delta/moe-chain.yaml) uses the backward witness without output
accumulation. Setup materializes AB descriptors for every expert. Scalar decode
assigns each worker an expert and local tile. The external domain conversion
selects local K coordinates against the descriptor's encoded token base. External
TMA/MMA produces fragments for positive K; empty backward reductions instead
supply register zeros. A proposed two-stage output ring emits every tile. Two is
a logical binding requiring the source-derived budget/layout checks, not a claim
that native compilation allocated exactly two stages. Descriptor visibility,
native fragment mapping and the source sentinel/tail correction remain open edges.

[Retained scalar evidence](checks/moe-dense-result.json) executes the actual
scheduler methods after erasing decorators/annotations and replacing Int32 casts,
constexpr and tuple-size operations with their integer meanings.360 scheduler
domains/7704 queries cover both scenarios, strides, peers, empty groups, ties and
both shortest-axis branches;1176 backward K0 CTA-output instances remain present.
An empty expert table reproduces the unguarded group0 read.1080 coordinate checks
evaluate retained extension tuples and descriptor pointer expressions;304 table
slot addresses establish the selected128-byte key/slot layout. A repeated token
offset is a discriminating mismatch. The K0 epilogue's actual zero/load selection
is replayed on scalar values; it does not emulate native GEMM.

A quiescent phase model supplies17 counterexamples for the final scheduler state:
after all published items retire, waiting with the last publication's phase can
ask for a generation with no future arrival. Advancing to the next state satisfies
all ring drains in that model. For a worker invalid initially, the source publishes
[-1,-1] and leaves the second record unconsumed. This is a source/protocol finding,
not an observed GPU hang. The retained PipelineAsync tail explicitly expects the
next useful buffer and waits all stage buffers. It does not establish ordered
retirement of other UMMA pipelines or change the previous paired-wait premise.

The new rule guards and version1 projection pass schema/expression checks.
No CUDA/DSL compilation, GPU execution, measured performance, new Realisation or
Store admission occurred. The scaled metadata/device companions complete source extraction of this five-file family.
