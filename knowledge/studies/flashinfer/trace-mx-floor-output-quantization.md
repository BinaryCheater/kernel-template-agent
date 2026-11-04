# MX output scales from a floor anchor

When a GEMM feeds an MX consumer, the output scale recipe is part of the
numerical contract. Task-scheduled FlashInfer GEMM selects a scale from the
**floor** of the complete group maximum's exponent. It does not select the
least power of two that contains the maximum after division by the format limit.
The new [candidate](delta/floor_anchor_group_quantization@1/template.yaml) and
[outline](delta/floor_anchor_group_quantization@1/proof.yaml) retain that choice.

Source paths below are relative to the retained [tree](inputs/sources.json).
The representative is runtime `flashinfer/prims_ts/batched_gemm/gmem_c_resources.py`,
not the initialization/test-data helper in `batched_gemm_quant.py`.

| Decision and concrete source binding | Rule field or boundary | Evidence |
|---|---|---|
| `batched_gemm_config.py:1064-1098`: MXE4M3 or MXE2M1 output,32 elements per scale | `floor_anchor_group_quantization@1.output_format`; destination format fixes anchor256 or4; `c_groups` fixes32 | Format and group size are not independent tuning knobs |
| `gmem_c_resources.py:2465-2472,3170-3201`: clear sign/mantissa of group maximum, multiply by1/256 or1/4 | New `program`: s=2**(floor(log2(m))-b), b=8 or2 | Existing MXFP8 ceiling-scale rule cannot bind this choice; MXFP4 is also outside that old format menu |
| `:3203-3221,3603-3664`: positive global scale g multiplies local maxima before reduction; reciprocal multiplies g again to form the code multiplier | `pre_values`, `pre_cast`: preserve FP32 operation order, finite intermediates and positive g | It would be wrong to remove g from the code path because it already affected scale selection |
| `:2514-2600,2719-2745`: M16 warp pieces write scratch, partner warp contributes to the M32 maximum, barriers bracket readers | `pre_groups`, `pre_lifetime`; reduction/publication refinements remain separate | One warp's maximum is insufficient; batched write/read phases share barriers but retain scratch generations |
| `:3170-3201`: exponent-bit reciprocal; UE8M0 scale store | `pre_range`; reuse `power_of_two_scale_encoding@1.site=producer` only on already-selected normal positive powers of two | k>=-118 for MXFP8 or-124 for MXFP4 makes s normal; selected normal s and reciprocal are required |
| `:74-91,2812-2815,3741-3770`: packed E4M3 TMA conversion explicitly RN-SATFINITE; direct scalar E4M3 and E2M1 use distinct lowering calls | `pre_cast`: RN-SATFINITE logical code policy, each native route separately bound | Scalar/E2M1 lowering semantics are not inferred from the explicit E4M3 packed conversion |
| `:3170-3201`: s==0 sets reciprocal0; scale is still converted to UE8M0 | Outside this version's live-scale domain | The stored-byte zero conversion is not established by the retained local helper; zero/subnormal/reserved-scale extensions stay named |
| `:2748` onward and store helpers: scale axis/layout follow output orientation, not a universal row-major byte map | `pre_metadata`: semantic group identity; reuse independent packing/layout rules only after mapping the axis | SwapAB and SF-C coordinates must travel with payload ownership |

For a positive normal maximum of exponent k, truncation returns2**k and anchor
multiplication yields2**(k-b). Within the admitted normal domain the bitwise
reciprocal is exactly2**(b-k). UE8M0 exponent encoding then preserves s. This
proves the scale recipe and metadata value; the code cast still carries its
rounding/saturation error obligation.

At maximum500, MXFP8 chooses s=1 and finite-saturating E4M3 code448, reconstructing448.
A ceiling recipe chooses s=2 and code256, reconstructing512. At maximum7,
MXFP4 chooses s=1/code6 rather than s=2/code4. Both examples distinguish the
contracts; they do not rank their quality for other distributions.

The [logical check](checks/check_floor_anchor_quantization.json) covers2490
normal maximum/mantissa cases across both formats, exponent-byte decode,
negative codes, paired M16 maxima and global scaling. One exponent below each
admitted lower bound gives s=2**-127, where the source bit reciprocal produces
s*r=0.75 rather than1. Nonzero subnormal maxima can truncate to zero entirely.
The check records these exclusions and the zero multiplier branch; it does not
invent a live zero-scale decoder.

Pool reuse: [groupwise_low_float_quantization@1](../cutlass/delta/groupwise_low_float_quantization@1/template.yaml)
supplies the existing ceiling-MXFP8 and rounded-E4M3 NVFP4 recipes, not this floor
choice. [power_of_two_scale_encoding@1](../deepgemm/delta/power_of_two_scale_encoding@1/template.yaml)
supplies lossless storage after scale selection. The earlier
[grouped fused output trace](trace-grouped-promotion-and-fused-output.md)
retains BF16 gate/activation checkpoints; selecting this output recipe does not
remove them. Native lowering, supported target/configuration menus and physical
scratch/layout proofs remain realization obligations.
