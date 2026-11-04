# gsplat reusable choices

Status: extracted; pending independent review.

| Kernel need | Rule and useful choice |
|---|---|
| Variable overlap needs ordered tile consumers | [Count/emit bins](delta/counted_overlap_depth_binning@1/template.yaml): duplicate overlap records into prefix-owned ranges then sort tile/depth keys |
| Geometric projection discards records | [Stable two-pass packing](delta/predicate_stable_record_compaction@1/template.yaml): same predicate, complete payload, block-prefix ownership |
| Nonlinear projection or antialias blur | [Sigma-point moments](delta/nonlinear_sigma_point_moments@1/template.yaml) and [opacity compensation](delta/blurred_gaussian_opacity_compensation@1/template.yaml): explicit weights and inactive-floor mass condition |
| Record fields or symmetric matrices | [Field-major storage](delta/record_field_axis_storage@1/template.yaml) / [triangular packing](delta/symmetric_matrix_packing@1/template.yaml): exact coordinate maps and matching backward |
| Sparse requested pixels | [Bit membership/rank](delta/occupancy_bit_rank_addressing@1/template.yaml): unique positions recover packed addresses |
| Long ordered compositing lists | [Batch summaries](delta/ordered_transmittance_batch_summary@1/template.yaml): parallel partial work plus exact crossing-batch replay; [conservative priming](delta/monotone_prefix_bound_hint@1/template.yaml) can reduce work |
| Backward needs forward history | [Reverse replay](delta/reverse_transmittance_replay@1/template.yaml): final T, last contributor and identical predicates recover local state |
| Sorted weighted-depth pair loss | [Prefix moments](delta/ordered_pair_distance_prefix_moments@1/template.yaml): linear rather than pairwise work, with evaluated-depth ordering premise |
| Inactive optimizer groups freeze | [Selective state update@2](delta/selective_parameter_state_update@2/template.yaml): explicit clockless versus local active-clock policy |

Composition: projected records → count/emit depth bins → ordered shared batches → final-T checkpoint → reverse replay → keyed gradient reduction. World-ray evaluation changes the Gaussian workload map; it can still use the ordered summary interface. Forward-only full crossing-batch folding is a separate approximate mode. Fixed camera mathematics, SH ownership, relocation and multi-device organization are accounted for in [evidence and bindings](trace-additional-designs.md); [inventory](inventory.md) reconciles coverage. [Focused checks](checks/replay.py) and schema results cover the stated local obligations, without GPU measurements.
