# Proposed pool delta

| Need | Candidate | Useful choice |
|---|---|---|
| Fuse a dense chain while preserving later training consumers | [operator_fusion@2](operator_fusion@2/template.yaml), [outline](operator_fusion@2/proof.yaml) | Complete-feature locality and optional retained checkpoint |
| Reuse expensive adjacent interval endpoints | [adjacent_boundary_reuse@1](adjacent_boundary_reuse@1/template.yaml), [outline](adjacent_boundary_reuse@1/proof.yaml) | Serial carry or aligned subgroup exchange, with explicit seam correction |
| Omit only the zero pure-input second path of an affine piece | [piecewise_affine_hessian_elision@1](piecewise_affine_hessian_elision@1/template.yaml), [outline](piecewise_affine_hessian_elision@1/proof.yaml) | Keep mixed parameter and upstream derivatives |
| Freeze inactive sparse optimizer parameters consistently | [selective_parameter_state_update@1](selective_parameter_state_update@1/template.yaml), [outline](selective_parameter_state_update@1/proof.yaml) | Gate moments, weight and local step clock together |
| Bound a trainable lattice table while preserving explicit collision semantics | [bounded_lattice_parameter_lookup@1](bounded_lattice_parameter_lookup@1/template.yaml), [outline](bounded_lattice_parameter_lookup@1/proof.yaml) | Dense/tiled/hash coordinate map and matching many-to-one gradient pullback |

Source bindings: [grid trace](../trace-grid.md). Existing `indexed_gather@1` does
not represent a trainable sharing relation; `shared_resource_binding@1` requires
a permutation. Neither can stand in for the new map merely because all three
compute addresses. The fusion [migration](operator_fusion@2/migration.md) retains version1. No live Store admission is performed.

Explicit extraction gaps for stencil construction, inverse roots and runtime mechanisms
are named in the [inventory](../inventory.md). Source comparison and derivative
limits are in the [simplex/composition trace](../trace-simplex-and-composition.md),
and state-clock bindings in the [training trace](../trace-training.md).
