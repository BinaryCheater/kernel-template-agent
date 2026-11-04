# TransformerEngine provisional delta

| Proposed pair | Useful choice |
|---|---|
| [quantization_scale_state_transition@1](quantization_scale_state_transition@1/template.yaml) · [outline](quantization_scale_state_transition@1/proof.yaml) | Scale epoch, complete maximum domain and next-producer history slot. |
| [hierarchical_low_float_scaling@1](hierarchical_low_float_scaling@1/template.yaml) · [outline](hierarchical_low_float_scaling@1/proof.yaml) | Rounded local scale paired with a coarse inverse. |
| [quantization_candidate_error_selection@1](quantization_candidate_error_selection@1/template.yaml) · [outline](quantization_candidate_error_selection@1/proof.yaml) | Compare complete stored candidates under one declared error score. |
| [centered_moment_merge@1](centered_moment_merge@1/template.yaml) · [outline](centered_moment_merge@1/proof.yaml) | Count-aware centered statistics with the between-partition correction. |
| [reusable_group_partial_allreduce@1](reusable_group_partial_allreduce@1/template.yaml) · [outline](reusable_group_partial_allreduce@1/proof.yaml) | Repeated CTA-group reduction using two banks and four counter phases. |
| [carry_compensated_parameter_split@1](carry_compensated_parameter_split@1/template.yaml) · [outline](carry_compensated_parameter_split@1/proof.yaml) | Share FP32 master high bits with a BF16 model using a paired low-word carry codec. |
| [radix_threshold_selection@2](radix_threshold_selection@2/template.yaml) · [outline](radix_threshold_selection@2/proof.yaml) | Extend radix digits to four/eight bits and bind threshold tie identities. |

| [chunk_ready_collective_pipeline@1](chunk_ready_collective_pipeline@1/template.yaml) · [outline](chunk_ready_collective_pipeline@1/proof.yaml) | Publish completed matrix chunks between GEMM and collective communication. |

All eight pairs passed [schema, pair-reference and expression checks](../inputs/schema-check.json).
Bindings and witnesses are in [scaling epochs](../trace-scaling-epochs.md),
[quantization/transpose](../trace-quantization-and-transpose.md),
[normalization](../trace-normalization.md), and
[routing/transforms/state](../trace-routing-transforms-and-state.md).
Two focused abstract checks cover [barrier generations](../checks/check_reusable_group_barrier.json)
and the [parameter bit codec](../checks/check_parameter_split.json). Both scripts
pass Ruff. [GEMM/communication](../trace-gemm-and-communication.md) records the
new chunk-handoff binding and external provider boundaries. Native execution and admission remain separate.

Radix version 1 remains intact. Its projection into version 2 keeps key_bits and
all original rank/count/value premises, binds digit_bits=8 and tie_policy=any,
and supplies tie-emission compatibility from its old arbitrary-tie contract.
Four-bit packed counters and smallest-index ties add explicit binding obligations;
source/migration details are in the routing comparison.
