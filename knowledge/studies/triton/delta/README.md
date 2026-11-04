# Proposed pool changes

All candidates are pending independent review; shared rules and admission files remain unchanged.

| Identity | Proof outline |
|---|---|
| [block_scaled_instruction_operands@1](block_scaled_instruction_operands@1/template.yaml) | [Outline](block_scaled_instruction_operands@1/proof.yaml) |
| [blocked_thread_register_layout@1](blocked_thread_register_layout@1/template.yaml) | [Outline](blocked_thread_register_layout@1/proof.yaml) |
| [carry_isolated_bit_counts@1](carry_isolated_bit_counts@1/template.yaml) | [Outline](carry_isolated_bit_counts@1/proof.yaml) |
| [code_domain_group_requantization@1](code_domain_group_requantization@1/template.yaml) | [Outline](code_domain_group_requantization@1/proof.yaml) |
| [contraction_numeric_recipe@1](contraction_numeric_recipe@1/template.yaml) | [Outline](contraction_numeric_recipe@1/proof.yaml) |
| [count_bucketed_sparse_row_reduction@1](count_bucketed_sparse_row_reduction@1/template.yaml) | [Outline](count_bucketed_sparse_row_reduction@1/proof.yaml) |
| [cyclic_tile_reduction_partition@1](cyclic_tile_reduction_partition@1/template.yaml) | [Outline](cyclic_tile_reduction_partition@1/proof.yaml) |
| [descriptor_union_completion_count@1](descriptor_union_completion_count@1/template.yaml) | [Outline](descriptor_union_completion_count@1/proof.yaml) |
| [early_dependent_grid_launch@1](early_dependent_grid_launch@1/template.yaml) | [Outline](early_dependent_grid_launch@1/proof.yaml) |
| [encoded_group_scale_recipe@1](encoded_group_scale_recipe@1/template.yaml) | [Outline](encoded_group_scale_recipe@1/proof.yaml) |
| [expected_observed_scale_pair@1](expected_observed_scale_pair@1/template.yaml) | [Outline](expected_observed_scale_pair@1/proof.yaml) |
| [fp4_bf16_bit_interleave@1](fp4_bf16_bit_interleave@1/template.yaml) | [Outline](fp4_bf16_bit_interleave@1/proof.yaml) |
| [indexed_descriptor_row_transfer@1](indexed_descriptor_row_transfer@1/template.yaml) | [Outline](indexed_descriptor_row_transfer@1/proof.yaml) |
| [launch_cancellation_work_reuse@2](launch_cancellation_work_reuse@2/template.yaml) | [Outline](launch_cancellation_work_reuse@2/proof.yaml) |
| [load_compute_store_role_pipeline@1](load_compute_store_role_pipeline@1/template.yaml) | [Outline](load_compute_store_role_pipeline@1/proof.yaml) |
| [mixed_radix_materialization@1](mixed_radix_materialization@1/template.yaml) | [Outline](mixed_radix_materialization@1/proof.yaml) |
| [operator_fusion@3](operator_fusion@3/template.yaml) | [Outline](operator_fusion@3/proof.yaml) |
| [opposite_order_topk_merge@1](opposite_order_topk_merge@1/template.yaml) | [Outline](opposite_order_topk_merge@1/proof.yaml) |
| [ordered_async_engine_chain@1](ordered_async_engine_chain@1/template.yaml) | [Outline](ordered_async_engine_chain@1/proof.yaml) |
| [owner_routed_occurrence_copy@1](owner_routed_occurrence_copy@1/template.yaml) | [Outline](owner_routed_occurrence_copy@1/proof.yaml) |
| [spare_operand_slots_epilogue_alias@1](spare_operand_slots_epilogue_alias@1/template.yaml) | [Outline](spare_operand_slots_epilogue_alias@1/proof.yaml) |
| [striped_locked_partial_accumulation@1](striped_locked_partial_accumulation@1/template.yaml) | [Outline](striped_locked_partial_accumulation@1/proof.yaml) |
| [striped_snake_rasterization@1](striped_snake_rasterization@1/template.yaml) | [Outline](striped_snake_rasterization@1/proof.yaml) |

[Source decisions](../trace-design-families.md), [production decisions](../trace-production-kernels.md), [version pins](../inputs/pool.json), [executed witnesses](../checks/results.md).

operator_fusion@3 retains all @2 fields/obligations and adds interleaved_swiglu plus pair/rounding premises. launch_cancellation_work_reuse@2 preserves @1 guarded scope and adds speculative_nontrapping extraction while guarding every coordinate use; its native accessor premise remains open.
