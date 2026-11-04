# Quantization domains, scale hierarchy and shared input transport

When the same tensor feeds consumers with different reduction orientations,
separate its logical quantization domains from output byte order. One input load
can feed multiple encodings, but transposing already quantized values generally
cannot replace independently quantizing columns. All paths below are under
`transformer_engine/common/` in the [retained source](inputs/sources.json).

| Representative source and binding | Reusable choice or proposed delta | Condition/difference |
|---|---|---|
| `transpose/cast_transpose.cu:103–215`: 128 threads, input vector and transpose-store vector widths determine tile extents | `staged_relayout@1`, `vectorized_access@1`, same-scale cast fusion | One thread computes each code once; rowwise output stores it directly, transpose path reuses it via padded shared `[32][33]`. Scalar per-tensor scale is the same for both. Optional amax atomically combines original input magnitudes; one designated thread writes inverse scale. This is the selected current-scaling test path. |
| `transpose/rtc/cast_transpose.cu` and `cast_transpose.cu:220–346` | RTC aligned specialization versus guarded general tail | Legal vector widths and whole tiles are implementation gates; the source heuristic estimates load/store efficiency and active SMs. No measured optimum follows from selection. Both retain the coordinate identity `out_t[j,i]=out_c[i,j]` for this same-scale path. |
| `transpose/quantize_transpose_vector_blockwise.cu:170–524` and square sibling | Independent row/column FP8 scales; vector versus square maximum domains | Grouped absmax, scale, cast and relayout compose. A row/column scale can differ at the same input coordinate. Square block sharing changes the requested representation. FP32-scale FP8 pairs are outside the old NVFP4/16 or MXFP8/32 rule's pinned format menu. |
| `cast/mxfp8/quantize_mxfp8.cuh:35–60,273–331,475–535` | MXFP8 vectors of 32; optional 32×32 common maximum; E8M0 scale and reciprocal exponent | Native SM100 path loads tiles with TMA, computes rowwise/columnwise maxima and emits scale-index mappings. Group assignment precedes numeric conversion. Existing `power_of_two_scale_encoding@1` explains storing already defined powers of two, not the policy that chooses the rounded scale. |
| `cast/mxfp8/specialized/cast_bidim.cuh:33–61,70–112` | Register-retained 32-row tile feeds two quantizations without a second input stage | Row reduction uses shuffles, column reduction crosses warps through shared scalar partials. **Both data outputs remain row-major** here; columnwise describes scale membership, not physical transpose. Eight warps cover the 32-row band. This is storage/reduction reuse, not an automatic transpose fusion. |
| Same specialized header, `cast_rowwise.cuh` and TMA specialized relatives | BF16 bit-pattern computation of scale exponent; register-resident versus staged transport | For finite normal BF16 amax, mantissa threshold 97 distinguishes crossing the 1.75 boundary. Add/mask gives the exponent without first rounding `amax*(1/448)` in FP32. Range clamps and special values require exact binding. Keep this numeric scale-generation gap; a storage-only E8M0 rule does not prove it. Rings reuse `multi_buffer_rotation@2`; `specialized/state_counter.cuh:18–39` flips phase only on stage wrap. |
| `CuTeDSL/cast/mxfp8/quantize_mxfp8.py:5–17`, `cast/mxfp8/quantize_mxfp8_cutedsl.cuh:219–252` | Optional CuTeDSL backend implements the same representation/activation flags | Default 64×64 CTA chunk, 64 threads, two 32×64 stages. The bridge is optional and can decline to the CUDA backend. DSL layout/native instructions remain a backend binding; importing CuTeDSL is not part of this study. |
| `cast/nvfp4/core_nvfp4.cuh:72–111`, `quantize_transpose_nvfp4.cuh:790–800,922–933,1112–1161` | New [hierarchical_low_float_scaling@1](delta/hierarchical_low_float_scaling@1/template.yaml) | Coarse maximum sets global multiplier; local E4M3 scale is rounded **before** deriving the data multiplier. Decoder multiplies E2M1 code by the stored local scale and coarse inverse. Per-row outer maxima are a different domain from per-tensor maxima. |
| `quantize_transpose_nvfp4.cuh:1265–1301,1451–1508,1834–1898` | NVFP4 vector16 versus square16; row-scaled outer domain; alternate scale type | Square path reduces a 16×16 maximum before both orientations. Row-scaled outer maxima exclude square groups and columnwise output. UE5M3 scale type is separately toolkit-gated and outside the new E4M3 candidate. Columnwise RHT is a different input transform, not permission to share unchanged rowwise values. |
| `cast/nvfp4/quantize_4over6_kernel.cuh:105–125,241–325,433–485` | New [quantization_candidate_error_selection@1](delta/quantization_candidate_error_selection@1/template.yaml) | Build stored scales for nominal map4/map6, quantize both, score actual decoded packets, select codes **and** scale together. MAE/MSE and FP32 decoded/FP16 proxy arithmetic are explicit policy choices. Shared square scales need a shared whole-square decision, not sixteen independent row winners. |
| `cast/nvfp4/quantize_4over6_nvfp4.cuh:121–180` | Restricted realization menu for that error-selection rule | E4M3 metadata, compact layout, deterministic conversion, 16-divisible dimensions; no stochastic rounding. Scale range can reserve 256 instead of 448. RTC/Blackwell architecture-specific conversion is required by the selected source path. |
| `cast/nvfp4/core_nvfp4.cuh:114–128`, standard NVFP4 conversion call sites | Stochastic conversion consumes Philox packets separately from deterministic quantization | RNG consumption depends on thread/vector/branch order. Existing rounding and bit packing do not establish stream equivalence after fusion or retiling; retain RNG identity as a binding gap. |
| `transpose/cast_transpose_fusion.cu:208–263` and device fusion bodies | `operator_fusion@1` for activation/gradient maps; `split_combine_kernel@1`, add, for dbias scratch | Quantization consumes the specified activation or derivative value. Bias partials preserve the specified pre/post-cast boundary, then a second kernel reduces them. MXFP8 reuses its column pass for dbias even when that pass emits no columnwise codes. Removing that pass would remove required reduction work. |
| `cast/core/grouped_layout.cuh`, `grouped_tma.cuh:34–127` and grouped quantizers | Segmented work and storage; `rebind_transfer_descriptors@2` plus staged transfers | Equal/varying row/column shapes change tensor/work decoding, not quantization algebra. Device metadata can update a base descriptor without a host shape read. One-warp convergent descriptor publication uses release, with matching consumer acquire. Source descriptor table capacity is 64; strides/pointers must be 16-byte aligned. Concurrent reuse of global descriptor slots needs ownership. |
| `cast/fused_group_requantize.cu:50–115,315–385` | Decode MXFP8 to the prescribed BF16 intermediate, recompute column scales and emit columnwise MXFP8 without a global BF16 tensor | `operator_fusion@1`, format decode and staged relayout compose. This is **requantization**, not code-preserving transpose. Native scaled conversion and fallback have explicit rounding boundaries. Grouped scale base `group_start/32*num_cols` relies on the /128 padded contract. TMA output waits before shared reuse. |

## Coordinate and value witnesses

For cast/transpose, select a source tile coordinate with `nvec_in=4`,
`nvec_out=2`, warp-local tile indices `i1=5,j1=7`, subindices `i2=1,j2=3`.
The source value is at tile-relative `(11,31)`. It is saved in
`local_output_t[3][iter].elt[1]`, staged at shared `[7][5]`, and read by the
transposed owner as `out_t[31,11]`. This distinguishes the byte permutation from
merely swapping scale addresses. Bounds guard source and destination tails.

For independent group scaling, put an outlier 64 at `(0,0)` and values 1
elsewhere in a 32×32 input. Coordinate `(0,1)` belongs to a row group with max 64
but a column group with max 1. Its two scales, and generally its code values,
differ even though the input value is identical. Specialized MXFP8 can keep
both code arrays row-major; NVFP4/legacy cast paths may physically transpose.
The consumer must bind both the scale-domain map and the data-address map.

For nested NVFP4, the ideal cancellation `local_scale*coarse_inverse=a/6`
holds only before storing/rounding the local scale. The emitted code must use
the rounded scale. An error selector that scores ideal candidates but stores
rounded metadata can select the wrong packet. FP16 proxy scoring is recorded
as a separate score policy; no theorem equates its winner to FP32 decoded MSE.

The [three provisional pairs](delta/index.md) add epoch state, nested scale
representation and candidate-error selection. Existing rules explain transport,
coordinate relayout and reduction composition. Group/tail/native rounding and
RNG/descriptor ownership limits are explicit; no implementation audit, native
compilation or per-entry checker was added.
