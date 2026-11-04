# triton reusable design menu

Status: representative-design extraction complete; pending independent review.

| Kernel need / choice | Rule |
|---|---|
| When hardware accepts block-scaled operands directly, preserve code and scale as separate instruction operands | [block_scaled_instruction_operands@1](delta/block_scaled_instruction_operands@1/template.yaml) |
| When choosing how a tensor tile spreads over registers, lanes and warps, vary the hierarchy factors and their digit order separately | [blocked_thread_register_layout@1](delta/blocked_thread_register_layout@1/template.yaml) |
| When multicast MMA has several operand replication patterns, derive completion arrivals from the union of elected CTA issuers | [descriptor_union_completion_count@1](delta/descriptor_union_completion_count@1/template.yaml) |
| When launch overhead or independent dependent-grid prologue can overlap a predecessor epilogue, release scheduling early and retain a completion wait before dependent loads | [early_dependent_grid_launch@1](delta/early_dependent_grid_launch@1/template.yaml) |
| When rows are irregular but each row needs the same contiguous column window, use an indexed descriptor transfer instead of scalar gather loads | [indexed_descriptor_row_transfer@1](delta/indexed_descriptor_row_transfer@1/template.yaml) |
| When cancellation response extraction is separable from work dispatch, keep successful ownership as the only permission to use returned coordinates | [launch_cancellation_work_reuse@2](delta/launch_cancellation_work_reuse@2/template.yaml) |
| When compute should overlap both input movement and output stores, use separate load, compute and store roles with independently sized rings | [load_compute_store_role_pipeline@1](delta/load_compute_store_role_pipeline@1/template.yaml) |
| When a tensor-core engine can consume a prior engine copy in issue order, chain copy→MMA→next copy and wait once at the external observation boundary | [ordered_async_engine_chain@1](delta/ordered_async_engine_chain@1/template.yaml) |
| When an operand ring has two spare slots after next-tile prefetch, borrow them for output staging instead of a separate C allocation | [spare_operand_slots_epilogue_alias@1](delta/spare_operand_slots_epilogue_alias@1/template.yaml) |
| When many rows contribute the same parameter-gradient vector, choose a bounded number of striped partial slots instead of one partial per row | [striped_locked_partial_accumulation@1](delta/striped_locked_partial_accumulation@1/template.yaml) |
| When persistent/CLC tile traversal should remain near the previous strip endpoint, alternate the major-axis direction between narrow strips | [striped_snake_rasterization@1](delta/striped_snake_rasterization@1/template.yaml) |

Multi-CTA matmul: cancellation response ownership → shared decoded work descriptor for three roles → staged operand loads → scaled or ordinary tensor-core MMA → independent accumulator generations → sliced shared output staging. Descriptor layout derives completion arrivals; input, accumulator and subtile depths remain separate knobs.

Evidence: [source-backed design traces](trace-design-families.md), [initial concrete bindings](trace.md), [coverage inventory](inventory.md), [executed witnesses](checks/results.md), [retained sources](inputs/sources.json).

Every new/revised rule has proof.yaml beside template.yaml. Native memory visibility, numerical guarantees, compiler mapping and hardware progress are recorded as specific open obligations; schema validation and semantic witnesses do not claim to discharge them. No GPU experiment or commit was performed.

## Shipped production choices

| Kernel need | Rule |
|---|---|
| Make operand casts, TF32 rounding and intrinsic accumulation policy explicit | [contraction_numeric_recipe@1](delta/contraction_numeric_recipe@1/template.yaml) |
| Keep bounded exact top-k state while streaming blocks | [opposite_order_topk_merge@1](delta/opposite_order_topk_merge@1/template.yaml) |
| Encode with a retained scale while measuring the next dynamic range | [expected_observed_scale_pair@1](delta/expected_observed_scale_pair@1/template.yaml) |
| Fuse the actual shifted/clipped interleaved gate formula | [operator_fusion@3](delta/operator_fusion@3/template.yaml) |
| Schedule sparse rows by contributor count and enumerate bitmap members | [count_bucketed_sparse_row_reduction@1](delta/count_bucketed_sparse_row_reduction@1/template.yaml) |
| Interleave K tiles across independent partial owners | [cyclic_tile_reduction_partition@1](delta/cyclic_tile_reduction_partition@1/template.yaml) |
| Store instruction fragments or contiguous tiles through explicit digit maps | [mixed_radix_materialization@1](delta/mixed_radix_materialization@1/template.yaml) |
| Prepare FP4 code bits for packed BF16 expansion | [fp4_bf16_bit_interleave@1](delta/fp4_bf16_bit_interleave@1/template.yaml) |
| Select output format and scale-rounding recipe together | [encoded_group_scale_recipe@1](delta/encoded_group_scale_recipe@1/template.yaml) |
| Find group maxima before decoding all FP4 values | [code_domain_group_requantization@1](delta/code_domain_group_requantization@1/template.yaml) |
| Move expert occurrences across unique owner shards | [owner_routed_occurrence_copy@1](delta/owner_routed_occurrence_copy@1/template.yaml) |
| Count packed membership columns without cross-field carries | [carry_isolated_bit_counts@1](delta/carry_isolated_bit_counts@1/template.yaml) |

One routed expert composition: peer occurrence copy → ragged work descriptors → persistent/scaled matmul → complete split-K combine → shifted gate → encoded group scale and its matching storage layout. [Production field bindings and source evidence](trace-production-kernels.md) give the guards and distinguish unsupported combinations.
