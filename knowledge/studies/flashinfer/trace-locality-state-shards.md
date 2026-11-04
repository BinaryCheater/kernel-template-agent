# Match independent state rows to memory and execution domains

When state rows are independent and most bytes belong to that state, split
every slot/head by value row, place the halves in separate locality-domain
memory and execute each on its matching domain. Splitting allocation alone
cannot provide the traffic property if both launches run on arbitrary SMs.

[locality_matched_state_shards@1](delta/locality_matched_state_shards@1/template.yaml)
and [outline](delta/locality_matched_state_shards@1/proof.yaml) add this coupled
placement/execution choice, complete-key row ownership and all-shard join.
The concrete source is the [retained](inputs/sources.json) Rubin T1 GDN path,
not a native locality or bandwidth measurement by this study.

| Concrete source decision | Rule field / binding | Evidence |
|---|---|---|
| `gdn_decode.py:117-152`: allocate two[slots,HV,64,128] parts, permute to[slots,HV,2,64,128] | pre_allocation/pre_mapping | Domain d local row r means original value row64*d+r. Each half retains all128 keys; no cross-domain dot-product partial is missing |
| `green_ctx.py:445-546`: CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN allocation per domain, align part bytes, expose explicit domain stride and retained owner | physical allocation boundary | The tensor is not an ordinary dense[slots,HV,128,128] view. Granularity padding and outside-caching-allocator ownership are real parts of its physical representation |
| Same module`:349-402`: localityDomainId assigned to green-context SM groups, optional driver backfill | pre_execution | Matching allocation and execution IDs must follow the target resource contract. Backfill/driver affinity and resulting hardware traffic remain target obligations |
| `gdn_decode.py:154-205`: select pool[:,:,d], narrow V and output to same64-row slice | execution_mode=matched_domain_streams or ordered_current_stream | Both modes compute disjoint complete rows. Inputs/slot maps remain the same despite request order |
| Same function records main fork event, both domain streams wait, each records done, main waits every done | pre_order | Output/checkpoint generation is complete only after both shards; no dense state reduction is needed, but completion still joins both |
| `:92-115,154-183`: cached domain contexts, warmup before capture; eager B*HV<8192 uses current-stream launches, graph uses domain streams | mode/cost policy | Current-stream fallback preserves row semantics but supplies no execution-domain traffic promise. Source threshold/prefetch policy is not a measurement or universal speed boundary |
| `green_ctx.py:408-443,509-546`: allocation owner/finalizer, synchronized unmap and deferred release around capture | pre_lifetime | View, streams and graph users retain the same allocation generation; logical tensor deletion is not permission to unmap a still-used captured address |
| `gdn_kernels/gdn_decode_bf16_state_rubin.py:865-936`: V64 half accepted; tile_v32/64/128 and device/stride/layout specialization | pre_backend/pre_numeric | Concrete half-row kernel exists on source SM107 menu, rather than only a host slice wrapper |

For V-major delta state, decay, prediction and rank-one correction of value
row r depend only on that row and shared key/gates. Query readout also uses
only that row. Thus splitting V preserves the recurrence without exchange.
Splitting K would instead split each prediction/readout and need a reduction;
the source does not do that.

The native row kernel also has a concrete warp organization. Its header and
`:120-148` show four independent row partials across a16-lane group, then a
transpose-reduce: two exchanges at offset8, one at4, then2 and1. Final row
owner is2*bit3(lane)+bit2(lane). The add pairing matches four independent XOR
butterflies, reducing source shuffle operations from16 to5. This is a
Realisation of reduction_partition@1 with num_groups16,group_size8 and
assignment=strided_threads over the complete128-key reduction; preserve the actual add tree for bitwise claims.
Scalar/packed FMA order and state IO conversions remain separate native
arithmetic obligations. State cp.async tiles are issued at entry in groups of
four owned rows; their copy and compute partition coincide, enabling
thread-private wait/consumption rather than a new CTA-wide handoff.

The [focused check](checks/check_locality_state_shards.json) covers98304
padded-domain state coordinates and the same coordinates under shuffled request
slots. It compares256 FP32 transpose-reduce lane results with independent
four-value butterflies using cancellation-sensitive inputs, and enumerates six
worker completion orders plus a missing-join counterexample. It records the
source eager/capture dispatch policy. Actual SM/memory matching, backfill,
CUDA-graph event behavior and allocator lifetime are native target obligations.

Seed physical_storage@1 binds global buffer occupancy/lifetime but has no
locality-domain execution variable. Seed output_tiling@2 can own each64x128 state-write
slice (tile_m64,tile_n128,executor=cta,index_map=custom) after each row prediction
is complete and its element updates are exposed as an independent map; it does not couple its executor with the
physical allocation domain or fork/join generation. The new rule owns that
missing coupling. attention_value_axis_partition@1 shares attention probability
frames across output slices; its program is not this recurrent state-placement
algorithm.

This disposes the two-domain state placement/update mechanism. Training and
selective-state codecs have different dependencies and remain separate rows
in the [recurrent inventory](recurrent-inventory.md).
