# SM120 persistent schedules: capture before release and striped consumers

When two compute groups alternate items from one input producer, each group needs
a cursor into the **producer's original stage sequence**, not a fresh per-item
ring. Skipping another group's item advances index/phase without issuing a release.
This is the retrieval cue for
[interleaved_consumer_streams@1](delta/interleaved_consumer_streams@1/template.yaml).
Register capture introduces a separate lifetime boundary: shared inputs can retire
before their final MMA when all remaining operands have already been copied into
registers.

Scope is the cooperative and pingpong mainloops/order/output protocols in the
[three-file GeForce family](inputs/geforce-blockscaled-variants.json), building on
[shared dispatch](trace-geforce-dispatch.md). Paths refer to the retained
[manifest](inputs/sources.json). [The layout/host companion](trace-geforce-layouts.md) completes physical mapping
and host/reference source dispositions. Native implementation obligations remain.

## Reference binding and actual differences

The [partial composition](delta/geforce-schedule-chain.yaml) uses mixed FP4 A,
FP8 E4M3 B, E8M0 SF/vector32, FP32 accumulation, tile128×128×128,
M128/N256/K640/L1 and one persistent CTA. Its local item sequence contains two
output tiles and each item contains five input stages/four native K32 blocks per
stage. Ring capacity3 is a logical witness pending resource/layout derivation.
Cooperative execution uses both four-warp groups on each item; pingpong assigns
local item ordinals0,2,... to group0 and1,3,... to group1. These are ordinals within
each CTA's persistent sequence, not parity of the global output-tile number.

| Decision and source | Actual binding | Rule field or gap |
|---|---|---|
| Cooperative constructor:150–191, grid:1364–1390 | Eight MMA warps and one DMA warp,288 threads; static persistent scheduler with cluster1. Both compute groups visit every assigned item. | warp_specialization@1 producer_warps1/consumer_groups2/epilogue_role=consumers supplies role-count intent for this case. Its same_sequence obligation applies; issuing-thread election/native load incidence remains an implementation binding. No one-warp-MMA granularity mismatch is claimed here. |
| Pingpong:578–600,870–903,1117–1120,1240–1244 | Still eight compute warps, but only four consume an item. Group1 initially skips one scheduler item and K stages; each group then advances two scheduler positions and skips the other item's K stages. | New interleaved_consumer_streams@1 groups2/steps_per_item5. Old warp_specialization@1's all-consumer same_sequence program cannot select alternating item ownership merely by setting consumer_groups2. Resource role counts and this ownership mechanism are separate. |
| PipelineTmaAsync: cooperative:535–551; pingpong:578–594; `pipeline/sm90.py:489–510,730–745` | Consumer arrival count8 cooperative or4 pingpong. Single-CTA signaling selects lane0 of each warp, so all active warps release once. | Per-slot readiness/release interface remains required. Count the owning compute group, not all eight warps, in pingpong. Skipped stages must never release another owner's generation. Native completion/progress remains open. |
| Pingpong jump:454–464 | Add iterations to count/index, xor phase by wrap parity; tile-local reset_count leaves physical index/phase untouched. | New rule's generation invariant q mod S and floor(q/S) mod2. Initial skip g*K and consume-K/skip-K jumps recover q=(g+2*j)*K+k. Changing K per item requires another map. Checks above S8 validate cursor arithmetic only, not an extension of multi_buffer_rotation@2's S<=8 scope. |
| Cooperative:765–917; corresponding pingpong loop | Preload block0. In steady iterations, compute current register block then load the next; at the last block release shared storage and wait/select the next stage before executing the already-captured final MMA. | physical_storage@1 pre_lifetime applies to the last **shared** read, not final arithmetic use of the register copy. Logical register tracking checks that all shared reads have occurred at release and that no live register is overwritten. Native LDSM/copy completion and per-lane layouts remain open. |
| Hoisted last iteration: cooperative:909–976; pingpong corresponding tail | Final stage loads the next register block before current MMA only when next block index>0; no subsequent stage is loaded. Mixed FP4 shifts the current fresh fragment once at either the steady or hoisted site. | Same operand_bitfield_placement@1 binding. The source's cyclic next-block expressions and capture model preserve exact-once preparation for positive K. This discharges a logical fragment-order edge, not native load-to-lane mapping. |
| K0: cooperative:765–815 and hoisted tail; pingpong:901–958 and tail | Alignment predicate accepts K0; accumulators are cleared, but unconditional preloads and a hoisted final block-MMA loop remain. Initial consumer_wait may bypass with token1 rather than block. | Positive K is required for these source mainloops. K0 is not established by accumulator initialization. Host/descriptor empty-shape handling is also unresolved; no GPU deadlock or output result is inferred. |
| PipelineOrder helper:pingpong:467–479; `pipeline/sm90.py:864–955` | depth2,length2; stage0 arbitrates MMA and stage1 arbitrates epilogue. Barrier index=stage*2+group; initial phase frees group0's first turn. After MMA, signal peer stage0 and wait stage1; after output drain, signal peer stage1. | External order protocol permits MMA(item+1) alongside epilogue(item), while serializing each resource's item order. The new stream rule requires this protocol; it does not implement it or assign a measured overlap benefit. |
| Pingpong shared declaration:383–394 | order array declares two Int64 entries; PipelineOrder creates/indexes four barriers, maximum index3. The following sA field is aligned to1024 bytes. | Declaration alone does not account for all used slots. The layout/host companion shows backing alignment padding for stages1..16 in a scalar field-layout model; compiled offsets remain unverified. This is not proof of native out-of-bounds or corruption. Scope order_protocol_valid cannot be asserted solely from the two-slot declaration. Verify backing/padding or correct the declaration before admission. |
| Cooperative output:1029–1103 | Shared C ring counter persists across items. Commit then acquire limits outstanding source reads to S-1. No explicit TMA-store producer_tail occurs at item/kernel exit. | staged_async_output_store@1 can supply source lifetime when its drain obligation is met. S1 acquire already drains each group; S>1 may leave source reads pending. Final source-read retirement is unresolved for cooperative multi-stage output. Kernel termination is not treated as proof of that retirement. |
| Pingpong output:1170–1244 | Shared C ring index restarts per item, but output executes in stage1 order and producer_tail drains before signaling the peer. | Existing output ring interface applies conditionally with a valid order protocol; shared C cannot be reused merely because the MMA stage has been handed off. Destination write completion remains a separate observation dependency. |
| Alignment helper/host:cooperative:1461–1511,1613–1615; pingpong equivalents | Host checks a_dtype for both operands. Reverse mixed A FP8/B FP4/K16 passes that check but fails the same helper with the actual B type. | Concrete missing-B-alignment source guard; no backend fault is claimed. Tile menus constrain tileK, not the logical problem K, and do not repair this missing operand-specific check. |

The declared stage policy reserves output stages min(subtile_count,4), then derives
AB/SF stages from shared capacity with helper overhead. Mixed operands use byte
shared allocation and TMA internal types. Full byte/layout agreement, stage counts
and SMEM bounds await the remaining physical-layout trace; do not substitute an
estimated stage count or broaden old rule domains.

## Protocol evidence and limits

[Retained checks](checks/geforce-schedules-result.json) execute the source jump
method on1896 cursor cases, including multiple wraps.1296 striped item-stream
domains cover empty/odd final stripes, K1..9 and S1..8, with exact once-only
consumption and no releases on skipped generations. Resetting a physical phase
would misidentify the first group1 generation at K3/S2.

A four-slot abstract order model explores235 states over0..12 items. It preserves
separate global MMA/epilogue order and contains66 states where MMA of the next item
and epilogue of the preceding item are both enabled. This model assumes correctly
backed barriers; it does not prove native source allocation, scheduling or progress.
The source two-entry declaration/four-index demand remains recorded separately.

Sixty-eight register-capture domains cover one through17 K stages and1/2/4/8
blocks per stage. Every required shared fragment is captured before release;
every MMA uses the intended fresh register code once. Source next-block expressions
and scheduler skip counts are checked. K0, reverse mixed alignment and cooperative
output-tail findings are retained as source boundaries. No CUDA/DSL compilation,
GPU execution, Realisation, ProofSummary or Store admission occurred.

The layout/host companion completes source extraction and supplies the conditional
order-padding argument; native progress remains open.
