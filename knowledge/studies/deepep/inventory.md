# Design inventory

Entry points: deep_ep/buffers/{ep,bucket,engram,pp}.py; Python API in csrc/python_api.cpp and runtime JIT specializations. Device bodies are under deep_ep/include/deep_ep/impls. All superproject bodies retained.

| Group | Representative / differences | Disposition |
|---|---|---|
| EP direct dispatch | ep/dispatch.cuh; NVLink vs RDMA, notify/dispatch warps, cached slot indices | Destination quotient by rank; notification packed arrival count and padded expert prefix partially traced |
| EP hybrid dispatch | ep/hybrid_dispatch.cuh; rail scaleout followed by local forward | destination_quotient_transfer@1; full forwarding publication and finite channel capacity open |
| EP combine | ep/combine.cuh and hybrid_combine.cuh, combine_utils | Inverse source identity and weighted sum required; reduction_partition@1 lead, full mapping open |
| Deferred dispatch-copy/combine-reduce epilogues | ep/dispatch_copy_epilogue.cuh, combine_reduce_epilogue.cuh | Producer/consumer contract and cached expanded layouts open |
| Redundant expert weights/gradients | ep/prefetch_weights.cuh and reduce_grads.cuh, eplb layout | Copy fanout versus FP32 additive return; exact placement and lifecycle open |
| Bucket all-reduce | bucket/all_reduce/nvlink.cuh; rdma and hybrid siblings | Multimem shard reduction then multicast store traced; cross-transport chunk protocols open |
| Bucket reduce-scatter/all-gather | bucket/{reduce_scatter,all_gather}/{nvlink,rdma,hybrid}.cuh where present | Complementary ownership designs; source bodies retained, detailed schedules open |
| Engram gather/wait | engram/engram_fetch.cuh and engram_fetch_wait.cuh | partial_batch_issue_retention@1; per-context final flush ownership inspected; wait implementation open |
| PP send/receive | pp/pp_send_recv.cuh | Named remaining protocol gap |
| Indexing and shared communication | csrc/indexing/main.cu, comm/barrier.cuh, handle/bucket | Preparation and backend implementation gaps; no claims inherited from old DeepEP |

Reconciliation: these groups cover every immediate impls subdirectory and EP device body; layout and common helpers are dependencies rather than separate operations. This establishes a source-family inventory, not that every group is explained.
