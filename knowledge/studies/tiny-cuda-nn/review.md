# tiny-cuda-nn: reusable design choices

Source-design pass complete: five proposed Template/ProofOutline pairs. Pinned
inputs and coverage dispositions are in [study](study.md) and [inventory](inventory.md).

| Kernel need | Rule | Useful choice |
|---|---|---|
| Keep a dense chain's hidden state local while preserving training consumers | [operator_fusion@2](delta/operator_fusion@2/template.yaml) | Full-feature ownership and optional retained checkpoint; [v1 projection](delta/operator_fusion@2/migration.md) |
| Skip only the zero pure-input Hessian of an affine piece | [piecewise_affine_hessian_elision@1](delta/piecewise_affine_hessian_elision@1/template.yaml) | Preserve parameter-mixed and upstream derivative paths |
| Freeze inactive sparse optimizer state | [selective_parameter_state_update@1](delta/selective_parameter_state_update@1/template.yaml) | Keep weight, moments and local update clock consistent |
| Use bounded learned lattice tables | [bounded_lattice_parameter_lookup@1](delta/bounded_lattice_parameter_lookup@1/template.yaml) | Explicit parameter tying and collision-safe gradient pullback |
| Subtract neighboring expensive boundary evaluations | [adjacent_boundary_reuse@1](delta/adjacent_boundary_reuse@1/template.yaml) | Serial carry or aligned subgroup exchange with the correct endpoint jump |

One composition: for an FP16 width64 network and sample128 CTA tile, apply
output_tiling@2 over samples and the full64 hidden features, operator_fusion@2 to
each activated dense edge, then physical_storage@1 to the reused padded shared
activation buffer. Training binds checkpoint=materialize and retains each
layer's global context; inference binds none. Input/output layers that violate
the local implementation's width/layout boundary are separate GEMMs. The
[network trace](trace-networks.md) records actual ownership, capacity and rounding
premises. A feature-sharded CTA decomposition is rejected until its missing
exchange is supplied.

Evidence: [network trace](trace-networks.md), [grid trace](trace-grid.md),
[basis trace](trace-basis-encodings.md), [five schema/pair/expression checks](inputs/schema-check.json),
and [ten-case source address permutation check](checks/check_fragment_serialization.json).
The latter ran through its [script](checks/check_fragment_serialization.py), which
also passes Ruff. These are source/logical checks, without native kernel runs,
performance claims, Store admission or end-to-end replay.

Further source bindings: [simplex/composition](trace-simplex-and-composition.md),
[optimizer state](trace-training.md), [runtime/loss](trace-runtime-and-loss.md).
Explicit gaps include stencil/coordinate construction, BaseConvert lookup,
inverse-root convergence/refresh, whole-training fusion, launch-graph reuse and
unsupported generated/legacy paths. They remain gaps, with no implied admission.
Next library: Dao-AILab/flash-attention.
