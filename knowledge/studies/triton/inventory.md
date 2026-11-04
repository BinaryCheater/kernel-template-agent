# triton reconciled NVIDIA design inventory

Status: representative-design extraction complete; pending independent review.

| Design family | Representative source | Concrete disposition |
|---|---|---|
| Vector map/fused row softmax | 01-vector-add; 02-fused-softmax:84-175 | Power-of-two masked rows, local max/exp/sum, occupancy-bounded static row stride. |
| LayerNorm | 05-layer-norm:53-95,140-220,267-291 | Centered local statistics, saved mean/rstd, striped_locked_partial_accumulation@1 for parameter gradients. |
| Dropout | 04-low-memory-dropout:110-145; language/random.py:5,89-155 | counter_indexed_mask_replay@1 binds Triton Philox10, original offsets, float comparison and fixed invocation seed. |
| Attention | 06-fused-attention:48-113,249-489 | Online base2 state, earlier/diagonal causal ranges, LSE replay, output-gradient row checkpoint and derivative tile ownership. |
| Grouped/persistent GEMM | 08-grouped-gemm; 09-persistent-matmul | Actual virtual-worker menu, scalar group cursor, grouped-M tail mapping; pointer versus descriptor paths. |
| Scaled MMA | 10-block-scaled-matmul:178-242; Gluon11:124-174 | block_scaled_instruction_operands@1, logical K versus packed bytes, blocked scale digits and separate scale TMEM. |
| Dependent launch | 11-programmatic-dependent-launch:28-67 | early_dependent_grid_launch@1 release versus completion wait. |
| Explicit layouts | Gluon02:30-120,160-175 | blocked_thread_register_layout@1 with exact whole-block map; smaller physical-replica regime excluded explicitly. |
| cp.async/TMA | Gluon03:41-65,258-331; Gluon04:133-164 | Group completion and slot generation; destination-write completion before message publication, not source-read-only retirement. |
| WGMMA/tcgen05/copy | Gluon05:338-380; Gluon06:210-246; Gluon10:5-74,117-142 | Register versus TMEM accumulators, use_acc first generation, ordered_async_engine_chain@1. |
| Cross-tile persistence/roles | Gluon07:627-724; Gluon08:96-307 | spare_operand_slots_epilogue_alias@1 and load_compute_store_role_pipeline@1 with separate rings/resources. |
| TMA indexed transfer | Gluon09:190-277,355 onward | indexed_descriptor_row_transfer@1, replicated offset lanes, asymmetric bounds/alignment and completion. |
| CLC scheduling | Gluon12:67-184; Gluon14:738-870 | Guarded work ownership; conditional speculative-coordinate variant in tutorial12; all-reader control-slot retirement in tutorial14. |
| Convolution/im2col | Gluon13:959-1120 | implicit_patch_contraction@1 dummy-depth binding, exact pixel-box/stride/padding and finite zero-tail product. |
| Multi-CTA | Gluon14:144-164,362-520,835-1028 | CGA reduction lowering, two-CTA MMA, multicast, descriptor_union_completion_count@1 and striped_snake_rasterization@1. |
| External/lowering boundary | 07-extern-functions; third_party/nvidia/backend/compiler.py; ReduceOpToLLVM.cpp:45-169 | Libdevice delegation and selected CUDA lowering/resource gates; no emitted artifact or non-NVIDIA route claimed. |

[Full decision/binding traces](trace-design-families.md) account for variant differences and failed old-rule matches. [Initial trace](trace.md) retains the first concrete bindings. Each source family above was read through a representative mechanism, rather than closed by a directory listing or a possible match.

Remaining realization obligations concern native numeric rounding, visibility/proxy rules, asynchronous lifetime/progress and independent assurance. They are listed in [outstanding checks](checks/outstanding.md); they are not unread design families. No compilation/GPU measurement or implementation audit was performed.

## Shipped production package

| Design family | Concrete disposition |
|---|---|
| Streaming exact top-k / selected backward | opposite_order_topk_merge@1; selected-domain softmax; unique-support dense scatter. |
| Gated activation / scale observation | operator_fusion@3 shifted clipped pair; expected_observed_scale_pair@1. |
| Contraction numeric preparation | contraction_numeric_recipe@1 explicit mixed-type widening, target-correct TF32 rounding and intrinsic accumulation policy. |
| Ragged and persistent GEMM | runtime_work_descriptors@1 padded schedule; grouped raster; cyclic_tile_reduction_partition@1; guarded indexed descriptor transfer. |
| Sparse weighted reduction | count_bucketed_sparse_row_reduction@1 local row buckets, bitmap traversal, row subtiles and feature chaining; linear backward. |
| Low-float conversion | encoded_group_scale_recipe@1 explicit up/down/nearest; code_domain_group_requantization@1 code maxima and16→32 regrouping. |
| Hopper/Blackwell storage | mixed_radix_materialization@1 digit profiles/nibble-axis exchange; fp4_bf16_bit_interleave@1; blocked_scale_digit_layout@1 reused for128x4 scales. |
| Distributed occurrence routes | owner_routed_occurrence_copy@1 unique expert owner, injective destination rows and published peer generations. |
| Membership statistics | carry_isolated_bit_counts@1 progressive4/8/32-bit counters; tile partials plus integer atomics. |
| Diagnostic and host helpers | FPSan strided blocked map delegates external codec; specialization/profiling/roofline helpers add no distinct optimization mechanism. AMD-only routes excluded. |

[Production source decisions and exact field bindings](trace-production-kernels.md). Every row is backed by a read representative; remaining native assurance is separate from source coverage.
