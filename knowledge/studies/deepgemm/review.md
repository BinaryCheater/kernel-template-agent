# DeepGEMM reusable rules

| Kernel need | Rule | Useful choice |
|---|---|---|
| Multiply block-scaled operands without materializing decoded matrices | [block_scaled_partial_accumulation@1](delta/block_scaled_partial_accumulation@1/template.yaml) | Choose a raw-dot extent within a region where both scales are constant; promote each partial with its own scales. |
| Store already-defined power-of-two scale metadata compactly | [power_of_two_scale_encoding@1](delta/power_of_two_scale_encoding@1/template.yaml) | Encode during production or a layout transform; require the exact live-value domain and retain scale ownership. |
| Share two sparse consumers' input requests | [shared_index_union@1](delta/shared_index_union@1/template.yaml) | Choose split size and retain each row's original output slots. |
| Move a common scale through a ReLU reduction | [nonnegative_scale_hoisting@1](delta/nonnegative_scale_hoisting@1/template.yaml) | Choose scale placement; require nonnegative, group-shared scale and admitted rounding. |
| Join producer results across repeated launches | [epoch_tagged_completion@1](delta/epoch_tagged_completion@1/template.yaml) | Bind the actual producer count; tag each invocation and retire all users before reuse. |
| Compute finite routing coefficient normalization | [row_column_rescaling@1](delta/row_column_rescaling@1/template.yaml) | Preserve requested iterations, epsilon placement and independent row/column groups. |
| Select exact top-k from distributed candidate groups | [sorted_run_topk@1](delta/sorted_run_topk@1/template.yaml) | Sort small runs once; merge their heads with a shared score/identity order and advance only the winner. |
| Start consumer reads as individual producer fragments finish | [fragment_parity_publication@1](delta/fragment_parity_publication@1/template.yaml) | Bind fragment/read-group sizes; wait on required parity bits and protect reuse until all reads finish. |
| Present tensors in a contraction's canonical axis order without copying | [tensor_axis_permutation@1](delta/tensor_axis_permutation@1/template.yaml) | Permute extents and strides together; validate packed metadata independently. |

For SM90 FP8 K256 with per128 K scales, partition into two raw128-product partials,
then promote each with its own A/B scales. An N96 tile starting at column96 crosses
B's scale boundary128, so its two column regions use different B scales. The
[partial chain](delta/chain.yaml) composes partition and promotion. Four other partial
chains cover attention, MHC, routing and fragment-ready L2 consumption.

**Source study complete; pending independent review/admission.** The retained160-file
superproject and both direct submodules are reconciled. All71 discovered API registrations
and28 device source files have explicit trace assignments. Nine Template/outline pairs,
eight CUDA Realisations and five partial Chains validate; all22 listed check commands
pass. [Inventory, trace assignments and replay evidence](checks/README.md) provide the
entry points for independent review. No GPU kernel or external backend was executed.

The retained boundaries matter for reuse:

- Numerical policy: SM90 post-dot scaling differs from SM100 scaled MMA; BF16 rounding
  precedes several fused consumers and FP8 outputs; raw-score top-k can differ from
  rounded-activation tie ordering. Statistical stochastic-rounding assurance is open.
- Ownership: masked length does not imply preservation of capacity tails; full source
  metadata cannot use payload-ring indices; duplicate fixed routes conflict with the
  MegaMoE per-expert source-table premise.
- Source counterexamples: locality balancing can leave a queue unclaimed without
  stealing; standalone SM100 prenorm has host/device N-menu and empty-split gaps;
  auxiliary row-major SF returns and hidden splits have unsupported-domain witnesses.
- Device obligations: release/acquire visibility, graph tags, asynchronous ring capacity,
  resident-grid progress, instruction semantics and numerical tolerances remain open.
  External libraries and allocator error paths retain their stated boundaries.

These gaps are located in the linked family traces; they are not silently converted
into candidate guarantees. No existing rule identity is replaced and no Store admission
is performed. The catalog continues with Marlin.
