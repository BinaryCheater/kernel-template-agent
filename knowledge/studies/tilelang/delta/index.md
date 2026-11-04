# TileLang provisional delta

| Proposed pair | Useful choice |
|---|---|
| [structured_zero_compression@1](structured_zero_compression@1/template.yaml) · [outline](structured_zero_compression@1/proof.yaml) | Encode existing 2:4 support as values plus positions; upstream pruning is separate. |
| [softmax_probability_recomputation@1](softmax_probability_recomputation@1/template.yaml) · [outline](softmax_probability_recomputation@1/proof.yaml) | Reconstruct probabilities from unchanged scores and row log-normalizers. |
| [chunked_decayed_outer_product_state@1](chunked_decayed_outer_product_state@1/template.yaml) · [outline](chunked_decayed_outer_product_state@1/proof.yaml) | Local triangular contraction plus ordered boundary state. |
| [blocked_unit_lower_inverse@1](blocked_unit_lower_inverse@1/template.yaml) · [outline](blocked_unit_lower_inverse@1/proof.yaml) | Diagonal solves plus dependent off-diagonal matrix products. |
| [radix_threshold_selection@1](radix_threshold_selection@1/template.yaml) · [outline](radix_threshold_selection@1/proof.yaml) | Byte-radix rank refinement and exact greater/equal quotas. |
| [streaming_topk_filter@1](streaming_topk_filter@1/template.yaml) · [outline](streaming_topk_filter@1/proof.yaml) | Threshold-filtered scan with exact bounded compaction. |

All six pairs passed [schema/reference/expression validation](../inputs/schema-check.json).
Their outlines contain source bindings and distinguishing algebraic witnesses.
Native target/layout/cast checks and independent admission remain separate.
No per-entry checker or benchmark was added. See the [review](../review.md) for
composition and the short list of retained pool gaps.
