# NVIDIA scheduling and pipeline source bindings

All source paths below are relative to python/tutorials unless stated otherwise. Concrete target examples are CUDA SM90 (Hopper) for TMA/WGMMA and SM100-class Blackwell for tcgen05/CLC. They are source paths, not measured executions.

| Source and concrete binding | Existing rule / field | Evidence / remaining boundary |
|---|---|---|
| 08-grouped-gemm.py:45-101 | persistent_tile_scheduler@1 fetch=static_stride, grid=NUM_SM, broadcast=none | Compiled menu uses virtual worker counts 84/128 or actual SM count. NUM_SM is not guaranteed to equal hardware SM count |
| 08-grouped-gemm.py:119-170; counts [6,1,4], workers=4 | online_group_work_decode@2 search=scalar_forward,cache_shapes=false (provisional) | tile_idx increases by NUM_SM across group boundaries; last_problem_end retains cumulative prefix. Queries 0,4,8 map to groups 0,0,2. Work decoding does not itself guarantee tail-safe memory |
| 08-grouped-gemm.py:145-167 | output_tiling@2 plus source full-tile precondition | Non-TMA body explicitly loads and stores unmasked, despite ceil tile counts. Bind M,N,K to tile multiples; uneven *group tile counts* are supported, arbitrary matrix residues are not established |
| 09-persistent-matmul.py:281-287 | block_rasterization@1 order=grouped_m,group=GROUP_SIZE_M | Derive group start and min(actual remaining rows,G). Final short group uses its own height for both modulo and division |
| 09-persistent-matmul.py:296-350,354-372 | persistent_tile_scheduler@1 static_stride; output_tiling@2 index_map=custom | Work IDs p+j*NUM_SMS; grid=min(NUM_SMS,total_tiles). A/B M/N indices outside shape map to zero; K loads zero-fill; C stores mask actual M,N. Separate epilogue counter is a source compiler workaround, not an independent algorithm parameter |
| 09-persistent-matmul.py:60-73,91-94,398-435 | async_transfer@1, compute_instruction_selection@2 | CUDA + major>=9 gates TMA/WS; >=10 gates CLC. Descriptor variants differ from masked pointer loads; exact generated instruction and completion not inferred from feature availability |
| gluon/07-persistence.py:602-623 | physical_storage@1 + multi_buffer_rotation@2 vocabulary, conditional realization | A/barrier ring index=sequence%num_buffers; B index=sequence%(num_buffers+STEALB); barrier phase=floor(consumer/num_buffers)%2. Different rings must not silently share an index. MMA completion is beyond seed CTA-barrier event model |
| gluon/07-persistence.py:627-650 | physical_storage@1 level=smem,buffer=shared | num_buffers>=3; STEALB adds one B slot and requires BM=BK or BM=2*BK. Without stealing, separate C staging lives alongside operand rings |
| gluon/07-persistence.py:660-693 | persistent_tile_scheduler@1 + async_transfer@1, source-specific glue | Next tile prologue interleaves with previous drain. Producer/consumer sequences continue across output tiles; math accumulator resets independently. Final peeled next-tile work is predicated at the shown sites; native guard behavior remains open |
| gluon/07-persistence.py:695-724; BM=128,BN=256,BK=64,S=4,STEALB=true | physical_storage@1 lifetime + staged_async_output_store@1 (provisional) | Wait MMA; take output, split along N into two (128,128) regions, each occupies one (64,256) B-slot byte capacity. Use B slots producer%5 and (producer+1)%5, fence shared writes, issue two stores. Wait pending stores before next destructive load; drain at exit |
| third_party/nvidia/backend/compiler.py:24-40,76-96 | compute_instruction_selection@2 target boundary | Type-dependent minimum dot shapes and PTX-version selection. No exact generated PTX/SASS is retained because no compiler execution was requested |

## Worked grouped-M mapping

For 5 M tiles, 3 N tiles and G=4, IDs 0..11 traverse rows 0..3 column by column. IDs 12..14 map to (4,0),(4,1),(4,2). Using G=4 instead of actual final height 1 as the modulo/division base would duplicate or escape valid coordinates. The grouped permutation is independent of which persistent worker obtains the linear ID.

## Reuse and pool boundaries

Pinned reference-package rules: persistent_tile_scheduler@1 and block_rasterization@1. Pin seed output_tiling@2, reduction_partition@1, physical_storage@1, async_transfer@1, shared_memory_tiling@1, shared_resource_binding@1, multi_buffer_rotation@2 and compute_instruction_selection@2. Provisional study rules: online_group_work_decode@2 and staged_async_output_store@1 from CUTLASS. These names/versions supply the hardest choices rather than prose-only invented freedom.

The alias choice itself is physical_storage@1 buffer=shared/share_order=other_first: consumed B bytes precede C staging, then the next B generation follows source-read completion. staged_async_output_store@1 supplies asynchronous source lifetime and separate output-observation obligation. The source's split-N output uses two physical operand-ring slots rather than the contiguous u%stages ring in that rule; it is a conditional composition, with slot-selection/retirement glue still open. Do not claim the old rule directly proves this schedule. No new general Template is proposed without establishing the joint A/B/epilogue live-set invariant.
