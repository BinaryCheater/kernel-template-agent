# TensorRT-LLM: selected residual encoding

For a low-bit tensor where a subset of reordered groups deserves extra storage, [selected_group_residual_encoding@1](delta/selected_group_residual_encoding@1/template.yaml) stores independently scaled base and residual codes. The budget controls which groups expand; rounding and consumer pairing are part of the contract.

Worked example: four16-element original groups, last two repaired, become base0, base1, base2, residual2, base3, residual3. Using one shared scale for every component or interpreting WEIGHT duplication as ACT residual changes the representation.

Status: partial; paused by user direction. One proposed pair is retained; materially missing families and source acquisition remain unfinished. [Trace](trace.md) also distinguishes attention capability gates, single/multiCTA sampling, output specialization and capped rejection bias. [Inventory](inventory.md) names the unread families. [Logical replay](checks/replay.json) verifies unique expanded coordinates; native numeric conversion and downstream reconstruction remain open. [Source manifest](inputs/sources.json) retains used inputs outside Git. No GPU measurement or admission.
