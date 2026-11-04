# MegaMoE: buffers, dispatch and fragment readiness

Pinned paths are relative to [the retained DeepGEMM tree](inputs/sources.json).
This continuation covers Python weight preparation and symmetric-buffer lifecycle,
C++ allocation/views, the complete shared scheduler, and selected BF16/FP8 device
paths for dispatch, publication and combine. The device arithmetic mainloops and
remaining quantized packing/epilogue details are still a tracing queue. This is a
partial MegaMoE disposition; neither kernel has been compiled or run.

## Public boundary and storage ownership

`deep_gemm/mega/__init__.py:17–83` builds one symmetric allocation per rank, aligning
capacity to1920 tokens (LCM of the eight candidate M widths). One-rank execution uses
ordinary CUDA allocation and its own pointer; multiple ranks use PyTorch symmetric
memory rendezvous. A fresh allocation is zeroed, followed by a process-group barrier
and CUDA synchronization. Reusing `base` requires the same group object and sufficient
bytes but does not re-zero storage. Nonoverlapping invocation and completed cleanup
are therefore caller/lifecycle premises. The PyTorch rendezvous and remote-mapping
implementation are external to the retained library source.

`layout/sym_buffer.cuh:10–40` retains a local base and up to72 rank-relative offsets.
Mapping local address base+o to rank r yields remote_base[r]+o, provided every rank
uses the same layout and o belongs to that allocation. It is not an arbitrary tensor
pointer mapper. Different local virtual bases are expected. Concurrent base reuse or
mismatched rank layouts would invalidate the correspondence.

`csrc/apis/mega_moe.hpp:36–161` allocates workspace, input token/SF/routing arrays,
optional shared-expert intermediates, routed L1 and L2 rings and combine staging.
The compute APIs below it accept BF16×BF16 or FP8 activations with FP8/FP4 routed
weights, architecture major10, SwiGLU only. FP8 recipe is fixed `(1,1,32)`; shared
weights on the quantized path must be FP8. Shape/localization compatibility and
buffer byte capacity are checked, while symmetric pointers and routing values need
caller validity. The launch uses the full runtime/device SM count, not a restricted
SM subset. Locality and the complete launch/resource bindings remain subsequent work.

Shared experts concatenate their intermediate channels into one wide shared MLP.
Their L1 input tokens alias the original x, but FP8 shared SF uses a separate layout.
Their combined L2 output occupies one extra combine slot regardless of the number
of shared experts. This differs from routing each shared expert as a separate top-k
slot. Debug mode zeros the complete allocation after the call; the caller must
recopy inputs on every debug invocation.

## Decisions and bindings

| Source decision | Rule / field or gap | Evidence and remaining boundary |
|---|---|---|
| Python `:115–164`: interleave gate/up every8 output rows, then transpose L1 SF in4×32 groups; L2 only transposes SF | `staged_relayout@1` / prepacking partial match | This materializes a copy, unlike `tensor_axis_permutation@1`'s shared allocation view. Weight and SF transforms must agree; interleave and SF transpose do not commute. Tagged coordinate replay covers12 shape domains. |
| `layout/mega_moe.cuh:13–38`, API `:45–80`: candidate M8/16/32/64/96/128/192/240, ring capacity maximized over all candidates | Storage extent derived from schedule, not freely tunable | Allocate logical full-pool capacity from worst receive tokens plus expert alignment; actual ring from the scheduler's live-block bound and align1920. Exact asynchronous occupancy/capacity proof remains open. |
| Layout `:55–94,136–149,244–253`: signals, full source slot table and TokenSrcMetadata | `physical_storage@1`; distinct lifetimes | Source metadata is indexed by logical pool token and retained for combine. Payload/token weights use ring indices. Ring-indexing metadata can overwrite source ownership while its result is still pending. |
| Layout `:342–452`, API input views | Explicit storage / packed scale ownership | FP8 input SF is contiguous `[capacity,H/128]` int32 (four K32 codes/word). Ring SF is MN-major with independent row capacity. Shared SF worst-case row capacity is ceil(capacity/8)*128. No BF16 SF arrays are allocated. |
| `heuristics/mega_moe.hpp:88–126`: expected routed tokens plus sqrt margin chooses M and store sizes | Dispatch heuristic, not runtime expert count | Two CTA clusters; BF16 K64, quantized K128. M240 quantized stores use L1 M24/L2 M8; BF16 uses M40. Remaining resource winner/launch replay is open. |
| BF16 device `:304–367`: count slot occurrences, reserve per-SM expert ranges, publish source indices and packed count | Runtime metadata generation, related to `runtime_work_descriptors@1` | Packed64 count has high32 arrival count and low32 token count. Final sum expects num_sms*num_ranks arrivals. Distinct expert IDs per token bound the per-source-rank table to capacity entries/expert. No duplicate-routing host validation was found. |
| BF16 `:398–488`: expert prefix and min-peeling round-robin source-rank decode | Grouped ownership / dynamic gather; new instance equations | For rank counts[3,0,1,2], order is (r0,s0),(r2,s0),(r3,s0),(r0,s1),(r3,s1),(r0,s2). Replay compares with explicit round-robin enumeration, including33/64/72 ranks. |
| BF16 `:497–552`, FP8 `:540–615`: pull payload/weight/SF, release full count, pad last token's contribution | `indexed_gather@1` partial; publication/lifetime separate | Physical payload row=logical_pool_token mod ring_tokens; full metadata keeps logical index. L1 reuse waits cumulative empty count. Last valid token contributes the remainder up to BLOCK_M; the counter denotes tile readiness, not number of valid rows. |
| `scheduler/mega_moe.cuh:16–79`: L1 warmup and live-pool estimate | Scheduling policy and storage/progress obligation | Warmup covers first L2 wave and L1/L2 task imbalance. Capacity adds frontier growth and one-wave margin. The closed form is source policy; its adequacy under actual asynchronous work stealing is not established by index replay. |
| Scheduler `:242–291,333–382,431–473`: per-expert prefixes, domain queues, optional stealing, claim after schedule-slot release | `persistent_tile_scheduler@1`, `per_slot_handshake@1` partial | Two-stage metadata ring. Locality splits whole N clusters; stealing only with at least two waves. L2 waits until required L1 tasks have been claimed in every domain, distinct from completed payload readiness. Moving claim before empty wait can reserve blocked work; preserve source ordering. |
| Scheduler `:140–172`, BF16 `:655–661,1013–1025`, FP8 `:723–730,1238–1250`: per-fragment release-XOR then grouped acquire | New `fragment_parity_publication@1` | One bit per L1 N128 block, giving64 post-SwiGLU features. BF16 K64 consumes one bit; FP8 K128 consumes two. Expected mask alternates by logical_pool_block/ring_blocks. All TMA output stores drain before publication. No-lap/read-lifetime premise remains independent. |
| BF16 `:885–890,1029–1036`: L1 output waits previous L2 empty; L2 announces input no longer needed | Buffer reuse protocol, not numerical completion | L1-input ring and L2-input ring have separate empty counts. L2 input can be released before its remote output store; full source metadata therefore cannot share that ring's lifetime. |
| BF16 `:930–968`, FP8 `:1110–1134`: gate/up BF16 round, asymmetric clamp, SwiGLU, routed weight | Numerical/fusion boundary, existing operator_fusion partial | Gate clamps above only; up clamps both signs. Weight is applied before L2 input BF16/FP8 conversion. Moving it after L2 changes rounding/quantization even though exact-real L2 is linear. FP8 amax/scale packing still needs the next detailed trace. |
| BF16 `:1193–1207,1217–1291`: tag peer completion and combine | Peer publication / ordered reduction, partial existing epoch match | Push own gridid+1 to peers during dispatch; later peer publishes readiness with destination's saved tag after local L2 writes. Consumer waits only selected routed peers. Shared output is added first, then active routed slots in slot order; BF16 inputs sum in FP32 then output rounds BF16. System-scope visibility, graph-id freshness/wrap and cleanup races need device evidence. |

## Concrete layout and lifetime witnesses

For an L1 tensor with N256 rows, original rows0..127 are gate and128..255 up.
Interleaving emits0..7,128..135,8..15,136..143,... . Its SF rows then undergo the
4×32 transpose in each128-row region. The output SF row j reads interleaved row
`128*floor(j/128) + 32*(j mod4) + floor((j mod128)/4)`.
The replay tags group,row,K-word separately; repeated byte values cannot conceal a
row swap. L2 weights retain their row order while their SF has only the transpose.

A1920-token ring with M8 has240 payload blocks but240*128=30720 SF rows. Thus SF
row capacity cannot be inferred from payload rows. With M240 each payload block
needs256 SF rows, not240. Each source formula is injective within one ring generation;
reuse aliases only the next generation's same physical slot. Full metadata remains
at logical pool indices, so a later logical token at1920 cannot replace the earlier
token's source rank/token/slot merely because both payloads use physical row0.

The per-expert/per-source-rank source table has capacity entries for an expert,
assuming distinct experts per token. For capacity1920, two slots choosing the same
expert for every token create3840 source entries. Slot1920 of expert0 crosses into
the next source-rank row, or the next expert on a single rank. The full-pool bound also uses
min(topk,experts_per_rank), which depends on the same uniqueness premise. MegaGate's
fixed-routing branch permits duplicates, so these interfaces cannot be composed
without validating uniqueness or changing MegaMoE's metadata allocation. This is a
caller-domain gap, not evidence of a device run.

The fragment mask permits subset readiness: with four bits, observed0001 and expected
1111 allow the first one-fragment read although the full tile is incomplete. A duplicate
XOR erases readiness. If generation1 and2 are allowed to lap an outstanding generation0
consumer, the same parity can match a different payload. The new rule therefore requires
exact-once publication and protected reuse; merely matching parity never proves lifetime.
For intermediate256, L1 has four N128 fragments. The BF16 partial chain uses four K64
read groups; the quantized neighbor uses two K128 groups of two fragments each.

## Proposed delta and remaining work

[fragment_parity_publication@1](delta/fragment_parity_publication@1/template.yaml) and
its outline isolate one mask with independently consumable groups. Per-slot handshake
protects a whole slot; epoch completion joins every producer. Neither encodes the
fragment-to-read-group incidence or partial-read readiness. The source uses XOR parity
rather than widening either existing rule's contract. The [partial L2 chain](delta/moe-chain.yaml)
composes its grouped waits with K partition; the full ring-capacity/lifetime proof is external.

[Scalar replays](checks/moe_layout_schedule.py) and [retained results](checks/moe-layout-schedule-result.json)
cover weight/SF coordinates, source-rank interleave, independent SF capacity, publication
orders over four generations, and BF16 weight/clamp boundaries. They establish equations
and counterexamples; no GPU result, memory-order proof or performance measurement is
produced. Resume with the remaining BF16/FP8/FP4 device mainloops, quantized amax/SF writes,
shared-expert paths, full scheduling/capacity replay and locality dependencies before
assigning a final source-backed disposition to MegaMoE.

## Quantized mainloop and epilogue continuation

`impls/sm100_fp8_fp4_mega_moe.cuh:752–816` selects routed or shared descriptors by
phase. Routed weights can be FP8 or FP4; shared weights are always FP8. FP4 selects
`float_e2m1_unpacksmem_t` and tensor-map datatype16U4_ALIGN16B via
`csrc/jit_kernels/impls/runtime_utils.hpp:63–98`. Shared-memory descriptor addressing
uses the same one-byte element size on both branches, while the routed FP4 transaction
count is halved for global weight bytes. The descriptor code requires inner dimensions
aligned128 for this unpack mode. This is hardware-assisted format expansion; it is
not a software nibble-to-FP8 conversion. Actual nibble placement and encoded instruction
semantics remain an ISA/device boundary; no decoded-value check is claimed.

The MMA issuer (`:827–941`) makes distinct block-scaled descriptors for routed/shared
weights and swaps operands: weights are UMMA A, activations UMMA B. UMMA M256 spans
the two CTA N128 weight tiles; UMMA N follows valid token rows rounded16. K128 contains
four K32 scaled MMAs with matching scale IDs0..3. Both SF operands use UE8M0 with
K32 grains. UTCCP transfers128-row SF groups into dedicated tensor-memory columns;
the source passes SFB before SFA after swapping operands. First MMA clears the
accumulator, later ones accumulate. The two-slot accumulator ring is independent of
the input pipeline, and the final empty wait protects its barrier teardown. Existing
`compute_instruction_selection@2`, `per_slot_handshake@1` and scale-storage refinements
are partial matches; SM90 post-dot scaled promotion is a different numerical algorithm.

The epilogue (`:1006–1051`) stages TMEM loads into registers. At M240 it prefetches one
store block; other M sizes use no prefetch. The final valid block's load is fenced
before releasing TMEM, even when that block was prefetched before its numerical use.
A warpgroup with no valid rows releases immediately. Registers then preserve the
remaining values after TMEM reuse. The replay checks463 valid-row/store-size domains
for exact load/consume coverage and one final release; asynchronous instruction
completion and register capacity remain device obligations.

For FP8 activation output (`:1110–1182`), gate/up first round to BF16, clamp and apply
weighted SwiGLU in FP32. Each lane has two features per token. The helper call
`warp_reduce<4,true>` preserves lane residue modulo4: it exchanges XOR4,8,16,
not XOR1,2. Eight lanes therefore reduce16 features; the adjacent warp supplies16
more, yielding the K32 amax. Only the even warp in each pair and lanes0..3 publish
SF bytes. `get_ue8m0_sf_exp`/inverse are the already-traced exponent scale policy;
FP8 conversion follows scaling, rather than quantizing each warp independently.

SF addresses (`:1192–1217`) use `(kcode/4)*row_stride*4 + sf_row*4 + kcode%4`.
The second token in a lane pair is16 bytes later because transposed SF rows differ4.
The replay independently enumerates all row/code coordinates and verifies5376
nonoverlapping byte writes across M16/32/64/128/192/240. This includes the M240
L1 store24 branch; L2 uses store8 and retains BF16 output before remote combine.
Shared SF uses the same row transform with its independent shared row stride.
The new checks and [results](checks/moe-quant-locality-result.json) also retain the
[locality queue counterexample](trace-locality.md). No new Template is needed for these
source bindings. Remaining MegaMoE work is the complete scheduler/capacity replay,
launch/resource reconciliation and final variant accounting.

## Scheduling, resources and variant closure

The [capacity replay](checks/moe_capacity_localized.py) substitutes the source warmup,
frontier-growth and wave-margin formula into640 bounded geometries, including zero
blocks and both L1/L2 task-ratio directions. An independent synchronous-wave enumeration
tracks each block from its first L1 cluster to its last L2 cluster; the source bound
covers every observed peak (389 tight cases). This validates the stated synchronous
abstraction only. Independent scheduler skew, two metadata lookahead slots, stealing,
transfer latency and arbitrary completion order are excluded. The source's closed form
therefore remains an implementation/liveness obligation, not a general proved bound.
The previously retained locality balancer counterexample independently blocks a blanket
all-queue progress claim even when byte capacity is sufficient.

Both launchers specialize all workload dimensions, rank/expert counts, top-k, block
sizes, ring capacity, stages, clamp and fast-math flag. Quantized adds SF geometry,
independent L1/L2 store sizes and routed weight type. Both launch one512-thread CTA
per device SM, cluster size2, with computed dynamic shared bytes. Dispatch128 threads,
non-epilogue128 and epilogue256 are fixed roles. Input full barriers expect four
arrivals (two operand loaders across two CTAs), empty barriers one issuer; accumulator
empty counts both CTAs' epilogue threads. Task metadata full/empty barriers protect
two stages and are released after epilogue has copied the task. These bindings reuse
warp_specialization@1 and per_slot_handshake@1 provisionally; input stages can exceed
the old rule's stage domain and are not silently admitted as such.

BF16 register budgets are48/40/208 for dispatch/other/epilogue with at most64 local
experts, otherwise96/88/160; both weighted totals are64512 registers. Static totals
are source guards, not proof of dynamic register reconfiguration progress. Quantized
budgets and shared-memory struct checks likewise remain source constraints awaiting
compilation. Descriptor offsets and actual device SM residency have not been measured.

`comm/barrier.cuh` separates dispatch and epilogue grid counters. One designated SM
contributes 2^31-(num_sms-1), other SMs contribute1, so completing the grid flips a
high-bit epoch. The caller synchronizes its participating warp subset before/after.
Cross-rank barriers alternate two signal locations and signs, cycling target counts
num_ranks,num_ranks,0,0; optional grid prologue/epilogue follows the call sites.
These use release/acquire system operations and assume whole-grid progress. They do
not match MHC's explicit128-wide epoch counter. Timeout is a fixed cycle threshold,
not a measured60-second guarantee across clocks. Cleanup overlaps combine only after
local L2 publication and the dispatch/epilogue rendezvous; each rank's cleanup then
meets a final cross-rank barrier. Device memory visibility and failure recovery remain
open obligations.

| Source variant | Shared mechanism and distinguishing boundary | Disposition |
|---|---|---|
| BF16 routed, shared absent/present | K64 unscaled MMA; weighted BF16 SwiGLU input to L2; shared input aliases x and shared output uses one slot | Traced through dispatcher, scheduler, epilogue and ordered combine. |
| FP8 routed weights | FP8 activations, K32 UE8M0 scaling within K128 MMA; paired-warp amax and FP8 activation re-quantization | Traced with SF ownership/address checks; actual conversion/MMA numerical admission open. |
| FP4 routed weights | TMA expands packed global storage into one-byte shared operand representation; weight-byte transaction count halves | Descriptor/type/count branch traced; hardware expansion/encoding boundary open. |
| Shared weights on quantized path | Always FP8 even when routed weights are FP4; independent descriptor and SF stride; concatenated intermediate channels | Both phase-selected descriptor branches and shared readiness counters accounted for. |
| M240 quantized vs other M | L1 store24/L2 store8, one-block register prefetch; others use equal stores without prefetch | Byte/row coverage and final-TMEM-release order replayed. |
| Locality one vs two domains | Two only if both layers' N cluster counts divide evenly; local queues and conditional stealing | Binding traced; balanced-domain caller assumption contradicted by source witness. |
| Masked routes, empty expert, empty token workload | Negative indices skip dispatch/combine; expert zero counts produce no M blocks; no-result combine initializes zero | Metadata and zero-block equations accounted for; actual full-grid termination open. |
| Fast math, finite/infinite clamp, debug cleanup | Arithmetic implementation/compile flags and full-buffer debug zeroing | Source-bound choices retained; no numerical tolerance or replay-safe input preservation inferred. |

Final source-study disposition for MegaMoE: public/preparation/build/device families
and their distinguishing branches are traced, with the bounded checks linked above.
The full asynchronous capacity/progress proof, locality correction, GPU address/race
checks and quantized arithmetic validation remain open implementation obligations.
Nine study candidates are unchanged; fragment_parity_publication@1 owns the new
selective-readiness invariant. Continue the remaining library inventory with legacy
Triton, auxiliary kernels and remaining GEMM/configuration/epilogue branches.
