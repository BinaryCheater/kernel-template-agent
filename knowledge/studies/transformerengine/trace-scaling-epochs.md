# Scale domain and update epoch

When quantization has a scale state, identify both its reduction domain and
**which data epoch produces it**. A maximum reduction alone does not describe
whether a tensor consumes its own maximum or a prior history. Paths below are
relative to the retained source in [manifest](inputs/sources.json).

## Current tensor scale before cast/transpose

`tests/cpp/operator/test_cast_transpose_current_scaling.cu:91–160` allocates both
rowwise and transposed outputs. For FP8 it calls `nvte_compute_amax`, then
`nvte_compute_scale_from_amax`, clears the quantizer's amax pointer to avoid
updating it again, and invokes `nvte_quantize` on the same CUDA stream.
The result is `Q_c[i,j]=cast_fp8(scale*X[i,j])` and
`Q_t[j,i]=Q_c[i,j]`, with corresponding inverse-scale metadata. Non-FP8 output
uses scale 1. This test supplies a selected call chain, not a run in this study.

| Source binding | Rule/choice | Consequential premise |
|---|---|---|
| `recipe/current_scaling.cu:24–115,170–176`: zero amax, grid-stride vector input scan, block max, one atomic max per CTA | `reduction_partition@1`, `cross_lane_reduction@1`, vectorized loads and global combine | Domain is **all elements of the current logical tensor**, not rows or quantization blocks. 512 threads, vector width `32/sizeof(InputType)` elements, up to 65535 CTAs. Reset precedes atomic updates; alignment selects guarded loader variants. |
| Same file `:267–335`: one-thread scale kernel follows completed amax | Metadata producer/consumer phase boundary | The scale consumed by quantization must be derived after the entire current amax reduction. Same-stream launch order supplies that boundary here. A fused one-pass quantizer using a previous scale has different semantics. |
| `recipe/recipe_common.cuh:14–48` | `scale=max_fp8/max(amax,epsilon)` using round-to-nearest division; optional power-of-two multiplier by clearing mantissa bits | This is a **multiplier**. Its power-of-two rounding is downward for positive normal scales, distinct from upward rounding of a dequantization divisor. Zero/Inf/NaN amax returns 1; quotient overflow clamps to supplied finite cap. Extreme/subnormal behavior remains a numerical binding. |
| `cast/cast.cu:21–26` → `cast/dispatch/quantize.cuh:71–91` | Output metadata requests per-tensor scaling and columnwise data, selecting `cast_transpose` when activation fusion is absent | The enum is named delayed tensor scaling even when the caller has precomputed a current scale. The caller's dependency chain, not that enum name, establishes epoch. Columnwise-only output is rejected on this path. |
| `transpose/cast_transpose.cu:25–99` | Four-warps-per-tile cast/transpose; vector load/store menu selected by source heuristic | Legal tiles require row/column divisibility for selected vector widths. The heuristic estimates cache-line elements and active SMs; it is not a measured optimum. Native transpose coordinates and reciprocal output metadata are the next representative to follow. |
| `multi_tensor/compute_scale.cu:26–55,81–87` | Same scale formula batched over lists of amax/scale/inverse-scale arrays, via tensor/chunk work metadata | This kernel **does not reduce original tensor data**. Tensor/chunk scheduler addresses independent scale entries; each writes scale and reciprocal. Its unused noop argument does not imply state preservation. |
| Same file `:59–77,91–101` | BF16 amax to E8M0 inverse-scale encoding for E4M3 | Distinct metadata representation and rounding; not the same FP32 multiplier output or another global maximum. |

For E4M3 max 448, a tensor filled with magnitude 2 produces multiplier 224 and
inverse 1/224. One magnitude-64 outlier changes the **entire tensor's** multiplier
to 7. With forced power-of-two multipliers these become 128 and 4 respectively.
This distinguishes tensor-wide current scaling from row/block scaling and from
the opposite rounding direction of a dequantization divisor. These are source
formula substitutions, not GPU observations.

## Delayed history is a state transition

`recipe/delayed_scaling.cu:84–153` assigns one CTA per scale. It snapshots
`history[0]`, rolls history left with the snapshot wrapped to the last entry,
and clears output history slot 0 for a later producer. Reads precede in-place
writes. The scale policy selects either the most recent snapshot or the maximum
of the pre-update history; the old scale is retained when the selected maximum
is not positive finite. Finite positive maximum yields
`scaled_max/amax`, with `scaled_max` incorporating the caller's margin; quotient
overflow clamps to FP32 maximum.

For old history `[9,7,2,5]`, the saved current maximum is 9; updated storage is
`[0,2,5,9]`, while MAX policy considers all four old entries. With history
`[2,64,1,3]`, MOST_RECENT selects 2 and MAX selects 64. That choice can change
quantized values even if physical transpose and arithmetic are identical.
The bulk variant additionally ingests a reduction buffer for current history.

Current and delayed special-value policies differ: current invalid/zero amax
returns multiplier 1, while this delayed path retains the old scale. Noop flags
must preserve whichever metadata/output state their actual branch promises;
they cannot be treated as a free early-return added to every helper.

Closest pool matches explain reduction, staging and packed scale formats, but
`groupwise_low_float_quantization@1` supports only its pinned NVFP4/16 and
MXFP8/32 pairs. It neither admits this whole-tensor FP32 scale format nor
expresses a history epoch. The proposed
[scale-state transition](delta/quantization_scale_state_transition@1/template.yaml)
now records that epoch choice; representation/fusion relatives are compared in
[quantization/transpose](trace-quantization-and-transpose.md). Native exceptional-value
propagation, transpose mapping and framework placement of delayed updates remain
explicit follow-up bindings. No per-entry checker or benchmark was added.
