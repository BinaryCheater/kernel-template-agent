# Block-scaled convolution: output policy, windows and frontend

Source is `examples/python/CuTeDSL/cute/blackwell/kernel/conv/dense_blockscaled_implicit_gemm_fprop.py`
at the pinned revision. Together with [the input trace](trace-conv-blockscaled-inputs.md),
this accounts for the companion's source choices. It is source-extracted with the
implementation/numeric gaps below; no compiler, GPU run or admission occurred.
[Bindings](inputs/conv-blockscaled-open.json) retain the source/method spans and
[checks](checks/conv-blockscaled-output-result.json) retain discriminating witnesses.

## Caller and frontend

| Decision and source | Selected or neighboring binding | Disposition |
|---|---|---|
| Frontend:3997–4034 | NVFP4 A/B=E2M1, SF=E4M3FN/vector16; MXFP8 A/B=E4M3FN, SF=E8M0FNU/vector32. Output is the matching narrow format or FP16/BF16. Other input/output pairings and FP32 D are rejected. | Workload representation boundaries, not independent quantizer knobs. The source wrapper chooses FP32 accumulation; frontend acceptance alone is not atom/physical-layout acceptance. |
| Frontend:4085–4184 | Per-CTA M must be128. N64/128/192/256 support repeated tiles; other parent-admitted N widths serve only one tile. C must be a multiple of4*vector. NVFP4 tileK64/128/256; MXFP8 tileK64/128; both cluster-M shapes cannot exceed staged SF count. | Five positive format/tile bindings and separate alignment/format/M64/K192/multi-N/cluster neighbors checked using retained pure predicates. C192 with NVFP4 tileK256 demonstrates partial per-filter channels. Complete parent geometry and im2col-field guards are inherited. |
| __call__:982–1038 | SFA/SFB dtypes match; SFD is optional and, when present, matches SF dtype. gen_sfd is pointer presence, not D dtype alone. Bias dtype must match D; beta!=0 requires a residual pointer and adds the residual warp. | Caller specialization. A narrow D without an SFD pointer is not automatically the proposed block quantizer. Bind gen_sfd=True when selecting the new output rule. No unspecified pointer/metadata combination is declared valid. |
| Frontend:4044–4061 | Narrow block output rejects bias, residual and registered non-identity activation. A custom callable not in that registry can still pass, even x→x+1. | New quantizer requires identity_after_quantization semantically. Registry membership does not prove that premise for an arbitrary accepted callable; scalar guard acceptance is retained as a gap, not a successful kernel run. |

## SFB N windows

Source _make_sfb_tma:1142–1174 uses the same N192 reshape as the earlier block-scaled
GEMM: nested shape ((2,2),rest), strides ((x,x),3*x). The logical128-channel scale
chunks become overlapping256-wide transfer windows. For tile j, outer=2*j,
rem=outer%4 and rest=outer//4; the loaded base is
128*(rem%2+rem//2+3*rest). An odd j selects64 channels into that window. Thus j1
loads[128,384) and selects[192,384); j2 loads from384. N64 instead shares one128
chunk between two tiles. TMEM offsets at2833–2858 are two words for odd N192/N64
tiles. N128/256 are aligned whole-chunk cases.

10880 coordinate checks cover64/128/192/256 and17 successive tiles. The source AST
retains the actual reshape strides; generated TMA/TMEM encoding remains open.
These are source metadata-layout bindings. cross_buffer_operand_window@1's W<=C
operand-buffer domain does not supply the N192/C128 metadata reshape. No existing
pool field chooses this reshape; a general metadata-layout abstraction remains a
named gap, with the instance mapping retained for later reuse.

## Output quantizer and numeric boundary

| Decision and source | Mapping or policy | Rule and evidence |
|---|---|---|
| Epilogue:3332–3379 | First multiply accumulator by runtime alpha; for16-bit D optionally add bias and beta*residual in accumulator precision. Residual TMA uses its own ring, released after register read. | operator_fusion@1 linear_combination supplies the tile-local combination. Readiness/lifetime belong to input/residual pipelines. Alpha/beta are workload coefficients, not accuracy-neutral tuning knobs. |
| Epilogue:3417–3443 | Each logical group computes amax, raw_scale=amax/M_D*norm_const, casts the scale to SFD dtype and decodes that stored scale before scaling output. M_D is6 for E2M1 or448 for E4M3FN. | New [groupwise_low_float_quantization@1](delta/groupwise_low_float_quantization@1/template.yaml), encoded_scale_used. Source normalization is runtime; this version binds normalization1. Other normalization needs its own admitted instance/variant. |
| `python/CuTeDSL/cutlass/base_dsl/typing.py:2553` and `_mlir_helpers/arith.py:197–213` | Float conversion emits arith.truncf; E8M0 requests upward rounding, other narrow formats use default nearest-even. The host reference at4997–5009 also rounds E8M0 upward to a bounded power. | New rule format_policy. The scalar intended-policy model enumerates finite E2M1/E4M3 values and E8M0 powers, including ties. Actual native lowering is not executed. |
| Epilogue:3444–3452 | Multiply values by min(norm_const*rcp_approx(decoded_scale),FP32_max), then cast at3471–3472. Zero E4M3 scale leads to positive infinite reciprocal clamped to FP32_max; E8M0 zero group uses minimum positive metadata. | New rule zero_group_policy_supported and reciprocal_matches_admitted_policy. Scalar zero-policy checks pass; exact approximate reciprocal, native min/zero conversion and FP32 error remain backend obligations. |
| Epilogue:3390–3463; host allocator:4547–4598 | logical_divide fragments into SF-size groups; emit metadata only for in-bounds M/N coordinates. Host SFD is unswizzled row/group/batch storage; code output uses the shared D ring and TMA store. | New rule logical_group/metadata_identity require the actual thread fragment to match the logical row/channel group. Exact lane mapping and partial-group neutral fill remain backend obligations. Metadata allocation/packing is separate from the quantizer; bitstream_packing@1 is not assumed. |
| Epilogue:3471–3472 | Output cast precedes epilogue_op. For narrow SFD outputs the admitted op must preserve values;16-bit output can use the requested activation. | Numeric-order glue; old operator_fusion enum does not prescribe arbitrary activation or cast placement. Humming's affine integer range/zero-point quantizer cannot supply this low-float policy. |

The new version has an explicit finite-code domain: finite input, representable
nonzero metadata scales and scaled values within the finite output range. Saturation,
extreme underflow and NaN/Inf behavior remain a source-to-backend extension gap.
For example, amax7 with NVFP4 rounds scale7/6 downward and makes the maximum
normalized value exceed6, outside this candidate's range. It is not silently covered.

A discriminating in-scope group is [7.4,2.17,0,...] with16 values. raw_scale=37/30,
stored E4M3 scale=5/4. Using the stored scale gives second E2M1 code3/2; incorrectly
using raw_scale gives2. Both are legal finite code values, so this distinguishes
metadata/code agreement without relying on overflow.33 scalar quantizer cases
cover both formats, signs, zero groups and several exact metadata magnitudes;
nearest-even midpoint checks cover the enumerated code tables. These check the
intended policy, not arbitrary device rounding/error tolerance.

## Residual order, overlap and scheduler boundaries

For overlapping accumulators (num_acc_stage1, N256), the epilogue reverses subtile
order when consumer phase0. The residual producer's shadow phase at2634–2659 uses
the same decision.20 source-expression domains compare both retained expressions
with independent expected output order; a producer ignoring phase would pair the
wrong residual subtile with the accumulator. Buffer occupancy/progress remains a
separate pipeline obligation. Source early release at3324–3329 follows the raw loop
index floor(SF_TMEM_columns/epi_tile_n); physical_storage@1's inequality applies per
aliased region, but does not choose that region map or reverse traversal itself.

CLC reader arrivals at1858–1869 are32*(1+cluster_size*(10+has_residual)), because
this entry adds four SFA copy warps. One/two-CTA groups need352/672 arrivals without
residual,384/736 with it. launch_cancellation_work_reuse@1 can bind those counts,
subject to its existing failed-response query guard. A retained pure frontend case
also admits cluster16 with C256/tileK256 and enough output rows: it needs5152 readers
without residual, outside that candidate's participant upper bound4096. Keep that
as an explicit domain-extension gap; do not widen the old record silently.

Input counter/multicast visibility, exact stage-capacity progress, SFD fragment/lane
mapping, numeric extremes and CLC lowering/domain remain implementation obligations.
All source branches now have dispositions across this trace and the input trace;
these obligations do not turn into claimed device assurance.
