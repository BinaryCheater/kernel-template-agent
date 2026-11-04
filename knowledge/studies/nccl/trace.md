# Ring and transport decisions

| Source | Binding / decision | Rule / evidence |
|---|---|---|
| src/device/all_reduce.h:14-82 | P4,r0: send3, reduce/send2, reduce/send1, finalize0, forward3,forward2,recv1 | ring_reduce_scatter_allgather@1 chunk ownership; rank index is ring order, not necessarily communicator rank |
| same:35-42 | loopCount=P*chunkCount; align partial loop chunks to16/sizeof(T) | Tail/legal storage premise; nelem can be nonpositive for empty chunks, primitive must avoid payload access |
| same:61-65 | final recv-reduce-copy-send postOp=true | post_operation exactly once after all rank contributions |
| src/device/prims_simple.h:103-178 | sender credit consumed+NCCL_STEPS>=step+d; receiver ready>=step+d | credit_gated_circular_transfer@1 no-overwrite/no-stale-read |
| same:270-300 | barrier before post; sender system fence when payload/FIFO metadata stored | Visibility remains actual transport premise; volatile polling comment explicitly differs from a formal acquire |
| src/device/prims_ll.h:100-156 | repeated flag read, two flags per line, cleanup on wrap | Distinct realization, not a shared-memory barrier |
| src/device/prims_ll128.h:57-92,192-279 | flag lanes and line payload; scoped fence | Distinct mapping; no throughput claim |
| src/device/all_reduce.h:146-172 | split tree reduce/broadcast thread budget | Policy binding, not reduction correctness parameter |

Composition: channel/chunk partition -> ring partial accumulation -> completed chunk -> protocol visibility -> forwarding -> output on every rank. Rank arithmetic and FIFO algebra substituted by hand; no GPU check performed. Full concurrent realization still needs peer execution evidence.
