> Historical first-pass record. Current status and dispositions are in study.md, inventory.md and designs.md.

# NVIDIA design inventory

Reconciled setup.py selective_scan sources, csrc/selective_scan headers and mamba_ssm/ops/triton modules. CUDA dtype wrappers instantiate shared forward/backward bodies; ROCm branches excluded. External causal-conv1d belongs to the next assigned library.

| Group | Source / variants | Disposition |
|---|---|---|
| Selective scan | csrc/selective_scan common/fwd/bwd, reverse_scan.cuh; fp32/fp16/bf16, real/complex, variable B/C | ordered_affine_prefix@1; backward reverse ordered scan and gradient partials reuse partition, exact derivative storage/recompute remains open. |
| SSD / Mamba2 chunk decomposition | ssd_combined.py, ssd_chunk_state.py, ssd_state_passing.py, ssd_chunk_scan.py, ssd_bmm.py | chunked_decayed_outer_product_state@1 provisional plus ordered affine chunk carry. Within-chunk causal BMM and boundary-state contraction are distinct pieces. |
| Inference update | selective_state_update.py | Direct affine recurrence, per-batch/head state ownership, grouped B/C, tied-head broadcast and optional D/z epilogue. |
| Norm/activation fusion | layer_norm.py, layernorm_gated.py, k_activations.py, softplus.py | reduction_partition@1, centered_moment_merge@1 and operator_fusion@2 provisional where appropriate; gate-before/after-norm are separate contracts. |
| Mamba3 SISO/MIMO and rotary | triton/mamba3/{angle_dt,mamba3_siso_fwd,mamba3_siso_bwd,mamba3_siso_step,mamba3_mimo_rotary_step,grouped_head_reduction,...}.py | Distinct trapezoidal/rotary recurrence, cached boundary K/V, angle processing and head reduction. Source inventoried; exact mechanism decomposition remains open, not subsumed by Mamba2. Blackwell num_stages=1 gate recorded. |
