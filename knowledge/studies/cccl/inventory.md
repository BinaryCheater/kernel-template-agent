# cccl reconciled NVIDIA design inventory

Status: representative-design extraction complete; pending independent review.

| Design family | Representative source | Concrete disposition |
|---|---|---|
| Block/warp/device reduction | block_reduce.cuh; agent_reduce.cuh; deterministic dispatch/RFA | Actual algorithm menu, partial warps; fixed-order versus binned_floating_accumulator@1. |
| Scan/segmented/key scan | agent_scan.cuh; agent_scan_by_key.cuh; single_pass_scan_operators.cuh | ordered_prefix_lookback@1, fixed anchors, head_reset_scan_state@1, exact key-boundary carry. |
| Radix/onesweep/segmented sort | agent_radix_sort_onesweep.cuh; agent_segmented_radix_sort.cuh | stable_radix_digit_scatter@1, homogeneous full-tile shortcut; segment-local bins and valid stores. |
| Merge/comparison sort | block_merge_sort.cuh; agent_merge.cuh; dispatch_merge.cuh | merge_path_partition@1, consistent A-first ties, shared staging and selected-agent tile size. |
| Multiset operations | agent_set_operations.cuh:107-207,578-637 | duplicate_balanced_set_partition@1, star correction, source-specific payload and compact output prefixes. |
| Select/partition/unique | agent_select_if.cuh:660-900 | Stable selected occurrence ranks, direct/shared scatter; distinct-output stable rejects versus reversed single-buffer rejects. |
| RLE/reduce-by-key | agent_rle.cuh:353-515,1026-1078; agent_reduce_by_key.cuh | Head/suffix state, cross-tile neighbors, singleton exclusion for nontrivial RLE, final suffix flush. |
| Histogram | agent_histogram.cuh:295-399,646-675 | privatized_bin_accumulation@1 storage choice, local equal-bin run updates and initialized final atomics. |
| Global/batched/cluster top-k | agent_topk.cuh:503-646; block_topk_air.cuh; agent_batched_topk_cluster.cuh | AIR radix rank, bounded_candidate_materialization@1, resident_overflow_repeated_scan@1, exact tie quotas/index preference. |
| Map/find/adjacent operations | agent_find.cuh:150-220; agent_adjacent_difference.cuh:130-245 | monotone_first_match_pruning@1; inplace_neighbor_boundary_checkpoint@1; ordinary independent transforms. |
| Batched copy/memcpy | agent_batch_memcpy.cuh:204-305,595-650,1030-1123 | Aligned vector interior/scalar edges and size_class_cooperative_assignment@1 with preserved descriptor identity. |
| Segmented reduction | kernels/kernel_segmented_reduce.cuh:130-227 | Actual segment begin/end, empty-init policy, maximum-size-selected subgroup/warp/CTA bodies. |
| Thrust CUDA/generic routes | sort.h; scan_by_key.h; gather.h; generic/binary_search.inl,scatter.inl | Concrete CUB delegation and accumulator override; per-query binary search, permutation gather/scatter with alias constraints. |
| libcudacxx/PTX wrappers | generated/tcgen05_mma.h; cp_async_bulk_tensor.h | Operand/address/ISA menus retained; wrappers alone do not prove selected caller completion. |

| cudax cooperative groups/reductions | coop/__group/mapping/binary_partition.cuh; coop/__algorithm/reduce.cuh:60-345 | sparse_lane_rank_reduction@1 for noncontiguous/non-power-of-two warp members; CUB thread/warp/block delegation, root DSM cluster table and cooperative grid partial table with actual barriers. |
| cudax multidimensional memory | __memory/__copy_bytes/memcpy_batch_tiles.cuh; __fill_bytes/fill_bytes_mdspan.cuh | Mixed-radix tile enumeration, driver CUDA13 batched-copy/fallback copy, stride-sorted simplified driver fills; no invented device kernel. |

[Full decision/binding traces](trace-design-families.md) account for variant differences and failed old-rule matches. [Initial trace](trace.md) retains the first concrete bindings. Each source family above was read through a representative mechanism, rather than closed by a directory listing or a possible match.

Remaining realization obligations concern native numeric rounding, visibility/proxy rules, asynchronous lifetime/progress and independent assurance. They are listed in [outstanding checks](checks/outstanding.md); they are not unread design families. No compilation/GPU measurement or implementation audit was performed.
