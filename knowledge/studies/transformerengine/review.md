# TransformerEngine: source-design pass complete

| Kernel need | Proposed reusable choice | Main condition |
|---|---|---|
| Quantize with a current or historical scale | [Scale-state transition](delta/quantization_scale_state_transition@1/template.yaml) | Preserve the requested data epoch, full maximum domain and next-producer history slot. |
| Fit low-bit group scales into a limited metadata format | [Hierarchical scaling](delta/hierarchical_low_float_scaling@1/template.yaml) | Quantize using the rounded local scale and pair it with the coarse inverse. |
| Trade clipping against code resolution | [Candidate error selection](delta/quantization_candidate_error_selection@1/template.yaml) | Compare complete stored code/scale pairs under one declared group metric. |
| Combine row statistics across warps/CTAs | [Centered moments](delta/centered_moment_merge@1/template.yaml) | Preserve counts and the between-partition mean correction. |
| Reuse a fixed group partial-reduction workspace | [Repeated allreduce](delta/reusable_group_partial_allreduce@1/template.yaml) | Two counters and two data banks retire old waits/reads before reuse. |
| Avoid a separate FP32 master allocation beside BF16 parameters | [Carry-aware split](delta/carry_compensated_parameter_split@1/template.yaml) | Initialization, midpoint policy and every update use the same paired bit codec. |
| Choose smaller radix histograms or deterministic threshold ties | [Radix threshold v2](delta/radix_threshold_selection@2/template.yaml) | Packed counts fit; ordered tie emission matches key/value tie classes. |

| Overlap collective communication with ready GEMM chunks | [Chunk handoff](delta/chunk_ready_collective_pipeline@1/template.yaml) | Match chunk identity, publication, scratch lifetime and final stream join. |

For a two-orientation quantizer, share input transport while keeping row/column
scale domains independent. A padded shared transpose preserves the same codes
only on the same-scale path; grouped requantization decodes and re-encodes.
[Scaling epochs](trace-scaling-epochs.md) and
[quantization/transpose](trace-quantization-and-transpose.md) give the source
bindings. Eight Template/ProofOutline pairs passed [schema/reference/expression](inputs/schema-check.json)
checks; native execution and independent admission remain separate.

[Normalization](trace-normalization.md) records the new statistic/protocol pairs
and the finite interleaving check. [Attention](trace-attention.md) reuses merge,
permutation and transfer rules around external cuDNN graphs.

[Routing/transforms/state](trace-routing-transforms-and-state.md) records the
parameter codec check, radix migration and further reuse. Both focused abstract
checks and their Ruff checks pass.

[GEMM/communication](trace-gemm-and-communication.md) distinguishes local
event/counter protocols from external vendor implementations. [Inventory](inventory.md)
accounts for the full source-design scope. Remaining gaps are native layout/math,
provider publication/progress, special-value/RNG policies and unavailable device
implementations. No native execution or admission is claimed.

Advance to tiny-cuda-nn. CUTLASS remains partially studied and paused.
