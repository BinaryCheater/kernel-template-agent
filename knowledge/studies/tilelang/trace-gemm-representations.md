# GEMM representation and grouping designs

When a contraction is bandwidth- or wave-limited, separate three questions:
what operand values mean, how their bytes reach the MMA, and which worker owns
an output partial. The examples below add distinct representation/scheduling
choices; related tile-size/backend variants reuse the same mechanisms.

All paths are relative to `examples/` in the [retained source](inputs/sources.json).
This comparison records representative designs rather than auditing each example.

| Design and representative | Reusable decision | Existing rule or precise gap |
|---|---|---|
| Native block-scaled MXFP8: `blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py::mxfp8_blockscaled_gemm` | Pack four E8M0 factors in a uint32. Data loads each K tile; scale loads every `(4*sf_granularity_k)//block_K` tiles. A separate warp transposes SF shared layout, then signals MMA; SF TMEM survives between scale loads. | Scale granularity is representation metadata; refresh cadence is derived. Reuse `staged_relayout@1` for the cooperative transpose, storage lifetime for persistent SF and earlier block-scaled grouping knowledge. Native MMA applies the factors; this is not accumulator-side promotion. |
| Promoted FP8 accumulator: `gemm_fp8/example_tilelang_gemm_fp8_2xAcc.py::matmul` | Inner accumulator is periodically added to an outer accumulator then cleared; a short last group is promoted once. Update interval is `128//block_K` when `block_K<128`, otherwise 1. | `reduction_partition@1` group_size and sequential combine explain the logical groups; two dedicated accumulator buffers supply the physical implementation. Both buffers may have the same declared dtype while native MMA behavior differs. Numerical regrouping/precision is a premise, not a guaranteed accuracy claim. |
| Software block-scaled accumulation: `deepseek_deepgemm/example_deepgemm_fp8_2xAcc.py` | Promote each scale-constant K partial with A's per-row factor and B's per-block factor into the final accumulator. | Direct `block_scaled_partial_accumulation@1` reuse, with source scale origins/group sizes. This differs from native scaled-MMA SF transport, not a new algorithm identity. |
| Integer dequantization: `dequantize_gemm/example_dequant_gemm_fine_grained.py` and W4A8 relative | Load packed B, unpack in local fragments, stage expanded B in shared memory for MMA. Unsigned extraction and signed conversion are distinct meanings. | Reuse `bitstream_packing@1` for fields and `staged_relayout@1` for staging. Old register-only `fused_dequantization@1` does not describe shared expanded B merely by name; the arithmetic/zero-point and converter layout must match. |
| Floating FP4/MXFP4 conversion: Hopper dequant examples | Floating payload/exponent conversion, optionally imported fast decode code, replaces integer unpack/cast. | `integer_operand_widening@1` is not a floating decoder. Preserve the representation-specific converter as a binding; a filename saying FP4 is not proof of standard E2M1/subnormal equivalence. No further format audit is needed for this survey. |
| Direct int4 MMA: `gemm_int4/example_tilelang_gemm_int4.py` | Declare packed int4 operands and consume them with `T.gemm`; host packs adjacent nibbles. | Same packing semantics with a native integer-MMA backend rather than floating widening. Frontend/example declarations do not establish compiled capability here. |
| BitNet prefill/decode: `bitnet-1.58b/kernel_benchmark/` | Prefill decodes packed/interleaved 2-bit weights for MMA; decode uses DP4A or scalar products followed by thread reduction. | Reuse offline prepack paired with its decoder, and `reduction_partition@1` for GEMV K ownership. This is a compute-organization difference selected by token count. The existing s4/s8/u8 widening candidate does not silently gain a 2-bit-to-int8 binding. |
| Structured sparse MMA: `gemm_sp/example_gemm_sp.py`, `sparse_utils.py` | Store two values per quartet plus position metadata and consume both in `T.gemm_sp`. Compressor retains all nonzeros and uses known zeros for unused retained slots. | New [structured_zero_compression@1](delta/structured_zero_compression@1/template.yaml), [outline](delta/structured_zero_compression@1/proof.yaml). Exact compression of existing support; upstream random/top-k sparsification is a different operation. Native metadata/lane layout remains a premise. |
| Block-sparse contraction: `blocksparse_gemm/example_blocksparse_gemm.py` | A per-output/K-block boolean mask suppresses both operand transfers and MMA. | Exact identity-only restriction when the operator's block mask defines zero contributions. Reuse the semantic mask-pruning obligation, not bounded approximate attention pruning. This is distinct from 2:4 compressed MMA metadata. |
| Pointer/segmented grouped GEMM: `grouped_gemm/example_grouped_gemm_fwd_ptr.py` and concatenated relative | Map a tile prefix to a group; either select raw A/B/C pointers or translate padded group-local M back to concatenated rows. | Reuse `segmented_axis_work_decode@1` and group boundary/tail guards. Pointer selection changes descriptor/address binding rather than the grouped contraction. Backward reduces the token axis per group and masks group tails. |
| Grouped native scaled MMA: `blockscaled_gemm_sm100/grouped_gemm_mxfp8_blockscaled_1d1d.py` | Expert offsets plus fixed grid capacity, two-CTA tiles, independent SF cadence. Extra peers load a clamped tile but only actual rows are stored. | Compose grouped decode, blockscaled representation and cooperative lifetime rules. This is not permission to skip a peer that owns required barrier participation. |
| Routed MoE: `fusedmoe/example_fusedmoe_tilelang.py` | Group token copies by expert, evaluate gate/up with fused activation/product, materialize the intermediate for down, apply route weights and scatter-sum duplicate token destinations. Shared expert uses the dense token path. | Reuse segmented work/padded storage and gather/scatter reduction ownership. Route weighting belongs after the expert nonlinearity unless an admitted transformation moves it. Sorting/prefix metadata is preprocessing, not a new GEMM algorithm. |

## New representation rule

The consequential sparse choice is whether known structured zeros are represented
explicitly or by a supported value/position encoding. For a finite quartet
`(1,0,-2,0)`, store values `(1,-2)` and positions `(0,2)`, nibble 8. Inserting the
missing zeros recovers exactly the same admitted dot product. A group with fewer
than two nonzeros fills the spare retained positions with known zeros. A group
with more than two nonzeros does not satisfy the rule.

The new pair covers this elementary logical identity and 16/32-bit metadata word
choice. It does not claim arbitrary dense pruning, TF32's 1:2 representation,
native metadata-layout validation or IEEE zero-times-NaN equivalence. No custom
checker was added: existing source compressor/consumer and the short proof settle
the design question; target support remains explicit. Candidate/outline schemas
were validated locally, with no compiler or GPU execution.

Scale transport, partial promotion and group decoding add concrete old-rule
instances. Unresolved format/native mappings are recorded once above and do not
block moving to the next distinct design. Next: attention/decode and recurrent
attention organizations, followed by reductions, convolution and lowering choices.
