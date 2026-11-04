> Historical first-pass record. Current status and dispositions are in study.md, inventory.md and designs.md.

# Causal-conv1d retrieval menu

- Continue a causal stencil across calls → [causal_window_state@1](delta/causal_window_state@1/template.yaml): history capacity, linear/ring layout and next-position contract.
- Full sequence with neighboring windows → shared_memory_tiling@1 plus physical_storage@1: exchange the left halo.
- Deterministic parameter gradients → reduction_partition@1: owned partial workspace followed by ordered combine.

Worked width3 witness gives [34,77,146,217] and survives chunking and ring wrap. [Trace/evidence](trace.md), [inventory](inventory.md), [check](checks/replay.md). Status: representative pass closed, one candidate pending review. Exact vector coordinates and initial/final-state gradient pullback remain named gaps. No GPU performance claim.


Continuation status (2026-11-04): reopened after depth review. Prior closure was premature: major distinct designs require concrete representative traces and field-level bindings. Existing proposals and checks are retained. Queue order remains QuACK → apex → Mamba → causal-conv1d.
