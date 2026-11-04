# Proposed Humming pool delta

Four new Template/ProofOutline pairs, five CUDA Realisations and two source-composition
chains are proposed at declared assurance. Starting identities and bodies are fixed by
[the pool snapshot](../inputs/pool-snapshot.json). The reviewed-study index was empty;
no pending candidate from another study is a dependency. No old version is changed.

| New rule | Choice the next Family can express | Invalid choice rejected | Evidence still needed |
|---|---|---|---|
| [bitstream_packing@1](bitstream_packing@1/template.yaml) · [outline](bitstream_packing@1/proof.yaml) | Continuous 1..8-bit codes and cross-word inverse, independently of matrix tile layout | Partial32-code group or an out-of-range code under the lossless premise | CUDA inactive-local initialization and compiled memory safety |
| [groupwise_affine_quantization@1](groupwise_affine_quantization@1/template.yaml) · [outline](groupwise_affine_quantization@1/proof.yaml) | Implement a requested groupwise unsigned affine quantizer and its zero-point policy | Group width, code width or zero-point policy inconsistent with the coarse contract | General FP32/CUDA rounding and group metadata coverage |
| [butterfly_linear_transform@1](butterfly_linear_transform@1/template.yaml) · [outline](butterfly_linear_transform@1/proof.yaml) | Register versus lane distribution of a requested normalized Hadamard block | Incomplete blocks, invalid lane grouping or a transform the workload did not request | Shared scratch races, exact compiled layouts and FP32/rsqrt tolerance |
| [first_writer_atomic_combine@1](first_writer_atomic_combine@1/template.yaml) · [outline](first_writer_atomic_combine@1/proof.yaml) | First-store publication then unordered atomic partial updates with a completion count | Wrong number of contributors, unsupported atomics or overlapping live generations | Memory visibility, TMA completion, floating-point order effects and progress |

Closest-rule comparisons:

- `offline_operand_prepack@1` introduces a matrix/fragment permutation with pack widths
  4/8/16. It cannot express G09's independent code-stream width or cross-word inverse.
- `fused_dequantization@1` decodes existing codes before a contraction. It does not
  introduce a requested quantizer's range, scale, zero point or rounding policy.
- `cross_lane_reduction@1` combines values into a reduction result. A normalized
  butterfly produces a full block of signed linear combinations with its own matrix obligation.
- `ordered_global_combine@1` serializes all partials in a fixed lock order. Humming's
  atomic path permits concurrent nonzero partials after the initializer; treating it as
  the old rule would incorrectly promise a deterministic order.

## Reused bindings and Realisations

[chain.yaml](chain.yaml) binds `offline_operand_prepack@1`, `output_tiling@2`,
`reduction_partition@1`, `physical_storage@1`, `fused_dequantization@1` and
`compute_instruction_selection@2` for ordinary A16/B4 GEMM. Its producer/consumer
interfaces, pattern parameters and ordering premises are in [bindings.yaml](bindings.yaml).
It is a **partial chain**: asynchronous schedule and A/output reconstruction are open,
and existing traffic effects overlap. It must not be priced as a complete kernel.

[quantizer-chain.yaml](quantizer-chain.yaml) composes the requested affine quantizer with
lossless code storage. Preserving codes does not recover the original real-valued weights.

[realisations/](realisations/) contains the four new rules' CUDA source routes and the
A16/B4 instance of `offline_operand_prepack@1`. They state source markers and scopes,
with no emitted SASS expectation or probed target claim. The prepack obligation is checked
through the MMA B layout for one explicit binding; other layouts are still gaps.

Additional conceptual reuse is recorded, with exact missing fields, in the family trace:
`block_rasterization@1`, `warp_specialization@1`, `register_rebalancing@1`,
`per_slot_handshake@1`, `linearized_work_partition@1`, `runtime_work_descriptors@1`,
`indexed_gather@1`, `cluster_multicast@1`, `smem_partial_combine@1`,
`ordered_global_combine@1`, `staged_relayout@1`, `shared_resource_binding@1`,
`vectorized_access@1`, `operator_fusion@1`, `operand_precision_conversion@1`,
`boundary_predication@1`, `split_combine_kernel@1`, `cross_lane_reduction@1`,
`async_transfer@1` and `multi_buffer_rotation@2`. A conceptual match is not a complete
legal binding when a field or obligation is missing.

## Open scope and migration

[gaps.yaml](gaps.yaml) contains fifteen schema-valid Gap bodies, affected groups and
specific next evidence. Most consequential: UMMA role/TMEM objects, MXMMA scale operands,
odd-width/native conversion domains and group/token scale factorization need additional
contracts. Candidate effects remain unpriced. These choices are not hidden in solver prose.

No split, merge, replacement or old-version domain change is proposed, so there is no
old-field migration. Existing versions and their historical examples remain addressable.
Independent review and admission are separate operations; neither the reviewed index nor
the live Store was modified.
