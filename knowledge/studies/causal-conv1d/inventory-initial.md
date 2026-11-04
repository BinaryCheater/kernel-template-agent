> Historical first-pass record. Current status and dispositions are in study.md, inventory.md and designs.md.

# NVIDIA family inventory

Reconciled setup.py sources, C++ pybind launch entry points and all three CUDA bodies. Four source translation units: causal_conv1d.cpp, fwd.cu, bwd.cu, update.cu. Dtype combinations and widths2/3/4 instantiate shared bodies.

| Design | Source / variants | Disposition |
|---|---|---|
| Sequence-contiguous forward | fwd.cu:47–171 | Shared vector exchange carries preceding thread/chunk halo. output_tiling@2 + physical_storage@1; zero left halo, bias/SiLU. |
| Channel-contiguous forward | fwd.cu:206–386 | Channel/length tile loads including left halo, initial/final state support and seq_idx predicates. Same stencil; distinct layout realization and segment boundary contract. |
| Streaming update | update.cu:24–136 | causal_window_state@1: linear shift or circular cache; state row indirection; padding negative index returns zeros. |
| Backward contiguous sequence | bwd.cu:54–293 | Reverse neighboring-gradient exchange; dweight partition and atomic sum. Gradient of SiLU applied before stencil pullback. |
| Backward contiguous channel | bwd.cu:329–644 | seq_idx/initial/final-state gradients; deterministic workspace or atomics selected by kDeterministic. State-gradient boundary mapping remains open. |
| Dispatch and workspace | causal_conv1d.cpp:153–259,265–459,467–end | Width/layout/dtype guards, deterministic scratch allocation, cached update parameters. Full library covered at design-group level; exact vector coordinate mapping and state-gradient pullback remain gaps. |
