# quack reusable design menu

| Kernel need | Candidate | Useful choice |
|---|---|---|
| Distributed vector reduction | [distributed_slice_reduce@1](delta/distributed_slice_reduce@1/template.yaml) | Subgroup/slice ownership and unique writers |
| Correlated weight codes | [cyclic_window_code_decode@1](delta/cyclic_window_code_decode@1/template.yaml) | Cyclic windows and consumer fragment coordinate map |
| Symmetric output | [symmetric_tile_mirroring@1](delta/symmetric_tile_mirroring@1/template.yaml) | Triangular group scheduling and mirrored stores |
| Stable narrow-float top-k | [value_lowbit_index_tag@1](delta/value_lowbit_index_tag@1/template.yaml) | Use spare mantissa bits for original indices |
| Large butterfly transform | [butterfly_bit_ownership@1](delta/butterfly_bit_ownership@1/template.yaml) | Logical bit stages, owner exchange and inverse stores |
| Stochastic BF16 conversion | [random_discarded_bit_rounding@1](delta/random_discarded_bit_rounding@1/template.yaml) | Random carry and per-element replay identity |
| FP4 without direct converter | [aligned_low_float_grid@1](delta/aligned_low_float_grid@1/template.yaml) | Align target grid with supported carrier |
| Ragged TMA sequences | [ragged_descriptor_rebase@1](delta/ragged_descriptor_rebase@1/template.yaml) | Synthetic bounds with canceled address translation |
| Ragged work lookup | [lookahead_prefix_group_decode@1](delta/lookahead_prefix_group_decode@1/template.yaml) | 31 lengths from32 boundary loads |
| Empty split-K participants | [reduction_partition@2](delta/reduction_partition@2/template.yaml) | Balanced ranges and neutral completion |

Existing rules are bound by actual fields in [the concrete design ledger](designs.md), including scoped exceptions where a seed is too narrow. Read that ledger for one composition, source locations, workload/backend constraints and variant differences. [Focused evidence](checks/designs.md) records executed checks. [Pinned sources](inputs/sources.json) and [pool versions](inputs/pool.json) are retained.

Status: source extraction complete after the depth continuation; proposals await independent review/admission. All materially distinct source groups have representative mechanisms and dispositions. Open native memory-order, backend-layout, floating-point and numeric-domain obligations remain explicit in the rules and ledger; no GPU performance or native validation is claimed. Historical first-pass records remain addressable with the `-initial` suffix.
