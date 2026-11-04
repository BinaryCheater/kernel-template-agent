# Representative ledger

Target NVIDIA CUDA, FP32/FP16/BF16 inputs with permitted weight type dispatch; arithmetic float in update.

| Decision / binding | Rule field or gap | Evidence |
|---|---|---|
| width3, w[0] oldest, w[2] current | causal_window_state@1 window | update.cu:69–110 loads two old samples, appends input and loops j increasing. No weight reversal is inferred. |
| H4,p5 reads slots3,0 and writes current at1 | causal_window_state@1 history_capacity, state_commit | update.cu:61–65,87–102. p is caller-owned cache_seqlens; device does not advance this array. |
| Linear history retains trailing H values by shift/append | Same logical boundary state, distinct realization | update.cu:71–84,95–97. Independent state row/channel owners are a premise even when conv_state_indices supplies indirection. |
| Negative state index writes output0 and returns | Inactive-row branch | update.cu:41–52. Does not touch state. Duplicate nonnegative indices cannot be assumed safe. |
| Sequence vector tile takes previous thread tail and keeps last thread tail for next chunk | physical_storage@1, shared_memory_tiling@1 | fwd.cu:75–104. Barriers before reading and overwriting exchange storage; halo exchange is not adjacent_boundary_reuse@1, whose domain is differences of a boundary function. |
| Channel-last initial state only on first length chunk; final state only on last | causal_window_state@1 instance, storage contract | fwd.cu:237–275. Segment id negative outputs0; unequal neighbor id contributes0 at304–331. IDs require contiguous segment identity; reused IDs within a window need distinct tags. |
| Width2/3/4 is source menu | width domain | causal_conv1d.cpp:176–193; initial/final states only channel-last. Read exact stride/alignment guards before backend reuse. |
| Deterministic weight gradient workspace vs atomics | reduction_partition@1 combine_complete, numerical order | bwd.cu:492–504 and launch kDeterministic; causal_conv1d.cpp:395–415. Final/initial-state gradient ownership still open. |

Worked zero boundary: x=[2,3,5,7], w=[11,13,17], no bias/gate gives [34,77,146,217]. Chunks [2] then[3,5,7] with correct history give the same outputs. A ring crossing p mod H=0 must preserve chronological order.
