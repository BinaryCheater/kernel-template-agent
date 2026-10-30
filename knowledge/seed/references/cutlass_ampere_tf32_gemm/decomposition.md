# CUTLASS Ampere TF32 tensor-op GEMM — decomposition

Source: NVIDIA/cutlass @ `e13f5cd4142c43efc37c2f6339923eb4a57afb99` (see `source.yaml`). Paths
below are relative to that commit. Configuration: example 14
(`examples/14_ampere_tf32_tensorop_gemm/ampere_tf32_tensorop_gemm.cu:170-226`): f32 A row-major,
B column-major, D row-major; CTA tile 128x128x16, warp tile 64x64x16, MMA 16x8x8, 4 stages,
`GemmIdentityThreadblockSwizzle<>`, epilogue `LinearCombination`.

Pool at decomposition time: assumed seed only (first reference). Status: analysis of source; not
built or run. ISA eligibility for GB10 (sm_121): mma.sync TF32, cp.async and ldmatrix are sm_80
instructions available on sm_12x, so `runnable_on: [gb10]` (eligibility, not verified).

## 1. Workload

`D = alpha * (A @ B) + beta * C` with A [M,K], B [K,N], C, D [M,N], all f32; alpha, beta runtime
scalars. The kernel rounds A and B to TF32 before multiplying, so the error contract must allow
TF32 operand rounding with FP32 accumulation. Operands must be 128-bit aligned
(`include/cutlass/gemm/kernel/gemm.h:182-196`).

## 2. Transaction inventory

| # | Class | Transaction | Where | Explained by |
| --- | --- | --- | --- | --- |
| T01 | schedule | grid = tiled shape; blockIdx -> (m, n) tile with log_tile = 0 | `include/cutlass/gemm/threadblock/threadblock_swizzle.h:110-139`, `include/cutlass/gemm/device/gemm.h:477, 495` | out_cta, raster |
| T02 | executor | 4 warps in a 2x2 arrangement, each a 64x64 warp tile; no K split across warps | `include/cutlass/gemm/threadblock/default_mma_core_sm80.h:1409-1411`, `include/cutlass/gemm/threadblock/mma_multistage.h:235-245` | out_warp |
| T03 | compute | K reduction in 16-wide k-tiles accumulated in place | `include/cutlass/gemm/kernel/gemm.h:229-235, 269-276`, `include/cutlass/gemm/threadblock/mma_multistage.h:649-662` | k_part |
| T04 | numeric | f32 -> tf32 operand rounding, round_half_ulp_truncate | `include/cutlass/gemm/warp/mma_tensor_op.h:355-380`, `include/cutlass/numeric_conversion.h:8362-8368` | tf32 |
| T05 | compute | mma.sync m16n8k8 TF32, f32 accumulate | `include/cutlass/arch/mma_sm80.h:217-260` | mma |
| T06 | storage | accumulators in registers (FragmentC) for the whole K loop | `include/cutlass/gemm/kernel/gemm.h:269-271` | acc_store |
| T07 | storage | A/B stage buffers in smem; mainloop storage aliased with epilogue storage | `include/cutlass/gemm/threadblock/mma_base.h:140-195`, `include/cutlass/gemm/kernel/gemm.h:139-141` | ab_smem |
| T08 | transfer | global -> smem cp.async.ca, 16 B, L2::128B prefetch hint | `include/cutlass/gemm/threadblock/mma_multistage.h:300-356`, `include/cutlass/arch/memory_sm80.h:151-170` | ab_async |
| T09 | transfer | thread map: 128 threads, 128-bit accesses, warp-raked | `include/cutlass/gemm/threadblock/default_mma_core_sm80.h:1425, 1459-1480` | ab_thread_map |
| T10 | transfer | zero-fill predication at M/N/K residue; mask clearing | `include/cutlass/gemm/threadblock/mma_multistage.h:310-315, 373-374, 591-592` | ab_guard |
| T11 | sync | commit_group per stage; wait_group<S-2> + __syncthreads | `include/cutlass/gemm/threadblock/mma_multistage.h:434, 486-491, 580-583` | ab_async (completion), ab_stages (barrier) |
| T12 | storage | 4-slot rotation, prologue prefetches 3 stages | `include/cutlass/gemm/threadblock/mma_multistage.h:250-284, 360-435` | ab_stages |
| T13 | schedule | cp.async issue split into groups interleaved between warp-MMA k-steps | `include/cutlass/gemm/threadblock/mma_multistage.h:551-576` | ab_async (issue_point) |
| T14 | access | crosswise XOR-swizzled smem layout for A and B | `include/cutlass/gemm/threadblock/default_mma_core_sm80.h:1447-1452`, `include/cutlass/layout/tensor_op_multiplicand_sm75.h:185-188` | ab_swizzle |
| T15 | transfer | smem -> register fragments with ldmatrix | `include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h:85, 410`, `include/cutlass/arch/memory_sm75.h:58-110` | frag_load |
| T16 | storage | register fragments double-buffered (load k+1 during MMA k) | `include/cutlass/gemm/threadblock/mma_multistage.h:177-183, 505-523, 598-604` | frag_stages (partial) |
| T17 | compute | epilogue alpha*acc + beta*C | `include/cutlass/epilogue/thread/linear_combination.h:239-247` | epi_fusion |
| T18 | transfer | C loaded in the epilogue | `include/cutlass/epilogue/threadblock/epilogue.h:488-491` | epi_fusion |
| T19 | transfer + sync | accumulators -> smem -> registers -> global per slice, barriers before and after the smem write | `include/cutlass/epilogue/threadblock/epilogue.h:484-535` | epi_stage |
| T20 | transfer | masked C loads / D stores at M, N edges | `include/cutlass/gemm/kernel/gemm.h:313-330` | d_guard |
| T21 | other | register allocation, unrolling, intra-iteration scheduling by nvcc | `include/cutlass/gemm/threadblock/mma_multistage.h:504` | none (ux-13) |

Excluded from the decomposed program: serial split-K with a global semaphore
(`include/cutlass/gemm/kernel/gemm.h:299-370`), compiled out because `SplitKSerial` is false in
example 14.

## 3. Refinement chain

From the workload down. M = matched against the pool before this reference; N = uses a Template
proposed here; P = partial.

| Step | Template | | Refines |
| --- | --- | --- | --- |
| out_cta | output_tiling | N | output map -> 128x128 CTA tiles |
| raster | block_rasterization | N | tile -> launch order |
| k_part | reduction_partition | M | K sum -> 16-wide sequential groups |
| out_warp | output_tiling | N | CTA tile -> 2x2 warp tiles |
| tf32 | operand_precision_conversion | N | f32 operands -> tf32 |
| mma | compute_instruction_selection | N | warp contraction -> m16n8k8 MMAs |
| acc_store | physical_storage | M | accumulator -> registers |
| ab_smem | physical_storage | M | A/B tiles -> smem, aliased with epilogue |
| ab_async | async_transfer | M | gmem -> smem copy -> cp.async issue + wait |
| ab_thread_map | vectorized_access | N | tile copy -> 128-bit thread accesses |
| ab_guard | boundary_predication | N | residue accesses -> zero-fill |
| ab_stages | multi_buffer_rotation | M | one buffer -> 4 slots, P = 3, beta = 1 |
| ab_swizzle | shared_resource_binding | M | smem addresses -> bank binding (XOR) |
| frag_load | collective_fragment_load | N | smem -> reg in fragment order |
| frag_stages | multi_buffer_rotation | P | one fragment -> two register slots |
| epi_fusion | operator_fusion | N | separate epilogue op -> fused |
| epi_stage | staged_relayout | N | register tile -> smem -> global |
| d_guard | boundary_predication | N | edge stores -> masked |

Match rate 6/18. 13 unmatched transactions (12 steps + T21), each with a Gap.

## 4. Least certain judgements

1. **frag_stages as a partial match.** One could argue register double buffering is not a storage
   rotation at all but an instruction-scheduling choice left to the compiler. It was bound to
   multi_buffer_rotation because the source explicitly allocates two fragment buffers and indexes
   them by `k % 2`; under that binding the seed obligation P <= S - 2 + beta fails for a correct
   program, which is the finding recorded in ux-09.
2. **ab_async absorbing the interleaved issue schedule (T13).** The interleaving of cp.async
   groups into the MMA loop was treated as a value of async_transfer's `issue_point`. If issue
   placement needs its own variables (group count per k-step), this should become a separate
   step.
3. **T01 as two steps.** The grid mapping and the launch order were separated (output_tiling vs
   block_rasterization) even though with log_tile = 0 the rasterization is the identity. Kept
   because the variable exists in the code and is non-trivial for other configurations.
