> Historical first-pass record. Current status and dispositions are in study.md, inventory.md and designs.md.

# QuACK retrieval menu

| Kernel need | Rule / useful choice |
|---|---|
| Reduce a vector but distribute final stores | [distributed_slice_reduce@1](delta/distributed_slice_reduce@1/template.yaml): subgroup size, slice size and unique owners |
| Split a long contraction | reduction_partition@1: separate workspace versus ordered or arrival-order finalization |
| Consume distributed inputs as shards arrive | provisional chunk_ready_collective_pipeline@1: shard-ready gate and progress contract |

Worked binding: eight lanes reduce four uniquely tagged slices; high XOR fold combines lanes0/4, then slice exchange at XOR2 and XOR1 leaves lane g holding slice g modulo4. Only lanes0–3 store. [Trace and evidence](trace.md); [inventory](inventory.md); [check](checks/replay.md).

Status: representative extraction pass closed; one new candidate pending independent review. Inventory includes named open layout, quantization and scheduling designs; those are not claimed fully explained. No GPU validation or measured performance. Source retained.


Continuation status (2026-11-04): reopened after depth review. Prior closure was premature: major distinct designs require concrete representative traces and field-level bindings. Existing proposals and checks are retained. Queue order remains QuACK → apex → Mamba → causal-conv1d.
