# DeepEP NVIDIA study

Status: partial; paused by user direction. Preliminary extraction only. Do not continue this study without a new request. No commits.

Retained input is the complete DeepEP V2.5 superproject archive recorded in [sources](inputs/sources.json), with MIT license. DeepJIT/NCCL dependencies are absent. This revision uses NVIDIA Hopper-or-newer/NCCL Gin and removes V1/NVSHMEM; old-backend descriptions cannot substitute for these bodies.

Two provisional pairs are saved: destination_quotient_transfer@1 and partial_batch_issue_retention@1. [Trace](trace.md) and existing finite [replay](checks/replay.json) cover limited dispatch locality/final issue observations. They do not prove transport visibility, complete combine or full-library coverage.

## Materially unfinished families and exact resume points

Resume only after sglang is completed and the user explicitly resumes the queue. Start at deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh around515-547 to finish forwarding publication and finite-channel capacity, then ep/combine.cuh,hybrid_combine.cuh and combine_utils for weighted inverse mapping. Complete dispatch_copy_epilogue.cuh and combine_reduce_epilogue.cuh layout/lifecycle contracts.

Further unread or partial distinct families are prefetch_weights.cuh/reduce_grads.cuh expert replication; bucket all_reduce,reduce_scatter,all_gather NVLink/RDMA/hybrid schedules and chunk protocols; engram_fetch_wait.cuh and context reset/progress; pp/pp_send_recv.cuh; csrc/indexing/main.cu and common barrier/handle preparation. The direct-dispatch packed arrival status around dispatch.cuh118-145 still needs carry/reset and readiness bindings. [Inventory](inventory.md) names every retained group; it is navigation, not completion.

All sources and artifacts remain preserved. No source expansion, GPU measurement or new closure check was performed for this pause checkpoint.
