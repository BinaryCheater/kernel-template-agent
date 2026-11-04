> Historical first-pass record. Current status and dispositions are in study.md, inventory.md and designs.md.

# Apex retrieval menu

- Many independent small tensors → [bounded_chunk_launch_table@1](delta/bounded_chunk_launch_table@1/template.yaml): bounded descriptor batches and chunk tails.
- Large group statistics → reduction_partition@1: complete a raw-moment workspace before normalization.
- Keep normalization in one launch → inspect the phase and residency gap before reusing one-pass synchronization.

Worked descriptor carry and statistics comparison: [trace](trace.md). [Coverage and unread designs](inventory.md). [Check](checks/replay.md). Status: representative extraction pass closed, one new proposal pending independent review; peer/DP/sparsity/v2 mapping gaps remain explicit. No GPU or performance claim.


Continuation status (2026-11-04): reopened after depth review. Prior closure was premature: major distinct designs require concrete representative traces and field-level bindings. Existing proposals and checks are retained. Queue order remains QuACK → apex → Mamba → causal-conv1d.
