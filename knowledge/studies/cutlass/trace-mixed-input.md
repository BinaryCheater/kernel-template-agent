# Blackwell mixed-input GEMM — batched conversion

Scope is the three kernel entries and `mixed_input_host_utils.py` in
`examples/python/CuTeDSL/cute/blackwell/kernel/mixed_input_gemm/`.
[Variant inventory](inputs/mixed-input-variants.json) retains all four files and
method spans. This trace source-extracts the batched entry `mixed_input_gemm.py`;
[the grouped companion](trace-grouped-mixed-input.md) completes both grouped kernels
and grouped host preparation. All four family files now have dispositions. Shared helper
paths below are relative to the pinned tree in inputs/sources.json.

## Batched contract and concrete branch

The witness uses M128/N128/K256/L1, Int4 A, BF16 B/C, FP32 accumulator, per-row A
scales shared over K256, MMA128×128×128, cluster1×1 and TMA output. A/B are K-major;
C is N-major. Convert-only neighbors use Int8/Uint8 with scale granularities0/0.
The public CLI restricts A to Int4/Int8/Uint8 and B to FP16/BF16; can_implement is a
weaker pure shape/scale/alignment guard, not a complete target capability query.

| Decision and source | Actual binding | Rule field or gap |
|---|---|---|
| Constructor:183–193; helper `utils/mixed_input_helpers.py:664–685` | Granularities0/0 select ConvertOnly; Int4 uses M1 and K granularity dividing total K and divisible by tileK. Witness groupK256 covers two K128 producer tiles. | Workload scale representation is fixed. Source predicates are replayed; arbitrary negative scale extent is not a supported binding merely because the guard accepts it. |
| Mainloop:1355–1369; helper:999–1024 | Convert integer A to B's compute dtype first, optionally multiply by the scale in that compute dtype, then store the transformed fragment. | New [integer_operand_widening@1](delta/integer_operand_widening@1/template.yaml), integer_identity and order_identity. Scaling is a separate operator_fusion@1 scale instance/numeric obligation. Moving it after accumulation is not part of exact widening. |
| `base_dsl/typing.py` Int4/Int8/Uint8 declarations | Int4/Int8 are signed two's-complement; Uint8 unsigned. All declared values are exactly representable in FP16 and BF16. | New rule input_format/compute_format. Marlin fused_dequantization@1's exact_conversion says q-8; raw Int4 code0 here means0, not-8. No method or zero_point label changes that old obligation. |
| Host utils:134–158; helper is_shuffle_a:688–709 | For K-major Int4→BF16, K divisible by8, scaleK>=8 and CUDA>=13.1, host stores eight values in[0,2,1,3,4,6,5,7] order. Device cvt_tensor_a pairs that order with cvt_i4_bf16_intrinsic. Otherwise natural typed conversion is used. | New rule conversion_path=paired_shuffle or typed; storage_order_matches_path and converter_path_available remain executable premises. CUDA gate is evaluated symbolically, not by probing an installed compiler. |
| `cute/arch/numeric_conversion.py:158–238` | Intrinsic documents the same eight-value permutation and restores natural BF16 order; wrapper also has smaller tails. | New version binds complete eight-value chunks only. Other fragment tails need a verified binding. Scalar permutation/code tests do not execute native converter instructions or prove the lane partition. |
| Helper:603–612,634–661; mainloop:1370–1388 | K-major transformed A feeds TMEM; MN-major feeds SMEM. Copy atom is selected accordingly, then transform threads synchronize and fence the appropriate memory view before readiness commit. | Source storage/instruction binding, not a freely independent location knob. physical_storage@1 can express per-region lifetime; it does not choose this major-mode/copy/lane mapping. Exact integer conversion produces register values before this stage. |
| Mainloop:1393–1401; MMA:1450–1492 | Narrow A load→transform, scale load→transform, B load→MMA and transformed A→MMA have distinct stage streams. Narrow inputs/scales retire after transformation; wide A/B retire after MMA consumption. | async_transfer@1 supplies abstract global→shared readiness; per_slot_handshake@1 and multi_buffer_rotation@2 apply only with legal bound counts/roles. They do not turn register→TMEM stores into global copies or justify an unbounded stage count. |
| Stage policy:1766–1919 | Accumulator stages derive from TMEM columns; TMEM A reserves3/2/1 accumulator stages at column thresholds128/256. Scale stages are0 for convert-only or4 for convert-scale. Load/transform stages derive jointly from aligned SMEM bytes, barriers and residual TMEM capacity, and each must be>=2. | Derived resource policy and coupling, not source knobs. C staging exists only for TMA stores and takes remaining SMEM. Actual CuTe layout/column counts and target occupancy remain backend obligations; no estimated sizes are presented as measurements. |
| Output:1953–1971 | Direct output requires M/N to divide the CTA tile; TMA output admits logical tails. The aligned M136/N144 neighbor passes only with TMA output. | output_tiling@1 plus staged_async_output_store@1 for the TMA path. Descriptor clipping and native store/source completion remain implementation premises. |
| Host utils:113–133 | Range-based quantization is computed, then scales are overwritten with random integers in[-3,3). The reference consumes those supplied integer codes/random scales. | Upstream test-data fixture, not a calibration algorithm or evidence for groupwise_affine_quantization@1. It is neither copied into product code nor treated as a numerical approximation to the initial random FP32 tensor. |

## Frontend and empty-work boundaries

The retained pure predicates accept the positive witness and reject Int4 without
scales, wrong row granularity, incompatible K grouping and direct-output tails.
They also accept scaleK=-128, and their cluster helper is not a complete target-capability
check. These acceptances do not make such configurations supported.

K0 also passes the frontend, but the actual retained get_divisibility(0) evaluates
log2(0) and raises during host input preparation. The MMA source sets ACCUMULATE=False
and writes raw accumulators only inside the K loop; there is no observed explicit
empty-loop identity write before producer_commit. The witness binds positive K;
empty-K initialization/preparation remains a source implementation gap.

## Replay and next work

[Checks](checks/mixed-input-widening-result.json) replay1056 integer-code/target-type
pairs,128 distinctive permutation packets, target-gate branches, scale/layout guards,
and output-tail neighbors. Ordinary FP16 packing and BF16 bit truncation independently
confirm exact integer representability; neither invokes CUDA. Source files and
converter/type declarations match the pinned manifest. The offset-binary mismatch
and an unpaired permutation remain discriminating counterexamples.

A prepack/decoder agreement is required, but offline_operand_prepack@1's fixed
Marlin interleave and constant-operand premise do not supply this dynamic A layout
by name. The new rule consumes the required order; upstream preparation cost and
wide-slot placement remain separate bindings.

[The grouped trace](trace-grouped-mixed-input.md) completes prefix search, hybrid
output and paired waits. Its accumulator-scale branch moves scales from
elementwise input conversion to FP32 partial promotion. The old partial rule
binds conditionally; a BF16 rounding counterexample excludes unrestricted
relocation. The helper's MXFP8/biased conversion functions have no selected call from
these three kernels; their presence is not a claimed backend for this family.
