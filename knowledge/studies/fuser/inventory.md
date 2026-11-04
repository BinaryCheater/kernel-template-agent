# fuser reconciled NVIDIA design inventory

Status: representative-design extraction complete. Sources and pinned revision are in [manifest](inputs/sources.json). The table is reconciled by distinct computation, layout, pipeline, quantization and scheduling against the named public registrations/tutorials and selected device bodies.

| Design family | Representative source | Concrete disposition |
|---|---|---|
| Pointwise | csrc/scheduler/pointwise_non_tma.cpp:30-183 | Thread output tiles, broadcast register reuse, vector/unroll policy; quantizer unroll1 constraint. |
| Block/grid/tuple reduction | runtime/block_reduction.cu; grid_reduction.cu; fused_reduction.cu | Partition/storage; designated_segment_completion@1; tuple component folds and double-bank retirement. |
| Persistent normalization | normalization_utils.cpp:845-933,1380-1433 | persistent_value_rematerialization@1 and register capture versus shared lifetime. |
| Outer Welford | runtime/block_welford_outer.cu:105-195 | centered_moment_merge@1; butterfly_reduce_scatter_chunks@1; cyclic_warp_row_skew@1. |
| Transpose | transpose_non_tma.cpp:248-271,457-494,627-633 | thread_ownership_exchange@1; independent vector/unroll factors. |
| Native matmul | matmul.cpp:68-88; matmul_hopper+.cpp | Hopper registers/WGMMA versus Blackwell TMEM/tcgen05, split-K, rectangular sections and role-specific pipeline. |
| CUTLASS/NVMMH integration | cutlass.cpp:68-95; nvmmh.cpp:24-95 | Explicit cc10.0 opt-in and external configuration-provider boundary; no external CUTLASS study reopened. |
| Scan/argsort/top-k | runtime/scan.cu, argsort.cu, topk.cu:85-130 | Inclusive CUB prefix; full radix order and top-k rank truncation; actual CCCL primitives studied next. |
| Block/grouped quantization | block_quantization_kernels.cu:149-419; codegen.cpp:1850-1905 | Group16/32 formats, supplied rounded scales, expert prefix/rebase and blocked_scale_digit_layout@1. |
| Communication/cluster | alltoallv.cu; cluster.cu:140-319; cuda_p2p.cpp:830-960 | Independent peer-byte routes, cluster_partial_destination_table@1, multimem_peer_address_operation@1 with caller handshake. |
| Resize/Greedy | resize.cpp:397-447; greedy.cpp:1016-1070,1341-1445,1670-1720 | pointwise_slice_demand_pushdown@1; constrained-op copies and actual shared exchange on conflicting thread maps. |
| NoOp/ExprEval/support | scheduler/registry.cpp:132-167; runtime helpers | No new numerical kernel from no-op/host registrations; wrappers followed through selected callers above. |

[Full decision/binding traces](trace-design-families.md) account for variant differences and failed old-rule matches. [Initial trace](trace.md) retains the first concrete bindings. Each source family above was read through a representative mechanism, rather than closed by a directory listing or a possible match.

Remaining realization obligations concern native numeric rounding, visibility/proxy rules, asynchronous lifetime/progress and independent assurance. They are listed in [outstanding checks](checks/outstanding.md); they are not unread design families. No compilation/GPU measurement or implementation audit was performed.
