> Historical first-pass record. Current status and dispositions are in study.md, inventory.md and designs.md.

# Mamba retrieval menu

- Input-dependent diagonal recurrence → [ordered_affine_prefix@1](delta/ordered_affine_prefix@1/template.yaml): ordered scan, identity tails and chunk carry.
- Matrix state with scalar decay → provisional chunked_decayed_outer_product_state@1: chunk-local contraction plus boundary state.
- One token inference → direct affine update with uniquely owned state row.

Worked maps(2,3),(5,7),(11,13) recover final state2125 from initial17. [Trace and evidence](trace.md), [inventory](inventory.md), [check](checks/replay.md). Status: representative extraction pass closed; candidate pending independent review. Mamba3 recurrence/rotary, full backward storage and exact norm layouts remain open. No GPU claim.


Continuation status (2026-11-04): reopened after depth review. Prior closure was premature: major distinct designs require concrete representative traces and field-level bindings. Existing proposals and checks are retained. Queue order remains QuACK → apex → Mamba → causal-conv1d.
