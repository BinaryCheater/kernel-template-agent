# Source choices and bindings

Source-relative prefix: deep_ep/include/deep_ep/. Concrete NVIDIA path uses NCCL Gin. All rule proposals are source-derived and unadmitted.

| Decision / location | Binding and rule | Obligation / result |
|---|---|---|
| impls/ep/dispatch.cuh:95–111 | Expert counts retain every expert; rank counts deduplicate expert/rank quotient. destination_quotient_transfer@1 | Topk<=32, expert IDs unique per token; invalid IDs excluded. Rank count is not expert count. |
| impls/ep/hybrid_dispatch.cuh:395–457 | expert / experts-per-scaleout maps to class; source ID=rank*max_tokens+token; class tail increments once; local bypass vs Gin remote put | destination_quotient_transfer@1 retains full top-k metadata. For8 experts, two classes of4, choices[1,2,6] send two payloads, then produce three expert inputs. shared_index_union@1 cannot express this transport class expansion; it merges sorted key sets with incidence maps. |
| Same file:515–547 | acquire-system signaled tail and TMA barrier phase protect forward reads | Separate transport visibility obligation, not discharged by quotient replay. Number of channels and forward chunk widths are compiled choices; topology-derived classes are fixed inputs. |
| impls/ep/dispatch.cuh:118–145 | 64-bit status packs arrival count high32 and contribution low32, wait high==numSMs, clear to0 | Reuse lead epoch_tagged_completion@1; packed addition/carry bound and readiness encoding differ, so no unconditional binding. |
| layout/ep/token.cuh:20–41 | Align hidden, SF and metadata individually, then RDMA frame, optional mbarrier suffix | Physical storage/local layout; alignment constants are backend contract, not free tuning knobs. |
| impls/engram/engram_fetch.cuh:109–178 | Compact peer requests into64-entry queue, B=32; issue only if pending>32; final nonempty issue rings doorbell | partial_batch_issue_retention@1. Sizes[32,0] leave32 for final; changing > to >= loses the explicit final issue. No claim that final issue completes transfer. |
| Same file:180–209 | Shared context final flush by last active issuer; release arrival then acquire fence | Completion and reset before next layer are separate constraints. No cross-epoch reuse proven. |
| impls/bucket/all_reduce/nvlink.cuh:30–64 | Each rank owns bucket_bytes/ranks; multimem load-reduce over float4 then multicast store, optional scale | reduction_partition@1 and reusable_group_partial_allreduce@1 retrieval; divisibility, rank ownership, initial/final barriers required. No RDMA equivalence inferred. |
| buffers/ep.py:431–541 | Analytic traffic/bandwidth heuristic chooses SM count; overlap preference changes floor | Dispatch policy, not semantic constraint or a measured result in this study. |

Composition: source incidence → destination quotient payload → remote publication → local expansion → expert computation → weighted combine by preserved source ID. Only the quotient refinement is proposed here; publication and inverse combine remain explicitly open.
