# Quantizer recipes and independent mask segments

When compressed masks have independently consumed rows or requests, preserve
segment boundaries at byte granularity. A global bit concatenation changes the
consumer's origins for every short segment. The new
[segmented_bit_packing@1](delta/segmented_bit_packing@1/template.yaml) and
[outline](delta/segmented_bit_packing@1/proof.yaml) capture byte order, short tails,
empty segments and separate packed prefixes. Paths below are relative to the
retained [tree](inputs/sources.json).

| Source choice | Concrete binding or boundary | Evidence |
|---|---|---|
| `quantization/packbits.py:130-143` computes per-segment ceil(length/8), cumsum, then Int32 dispatch | `segmented_bit_packing@1.bit_order=big/little`; original boolean and packed-byte prefixes are distinct | Empty segments repeat output starts; prefix arithmetic must fit before and after the Int32 conversion |
| `include/flashinfer/quantization.cuh:215-265` assigns one byte per thread,8 booleans, zero-default CUB loads,256 threads; one CTA loops over each segment | New `program`, `pre_ownership`, `pre_tail` | A segment's first value maps to bit7 or bit0 of its first byte; missing final-byte values become zero and empty segments write nothing |
| Existing `bitstream_packing@1` uses32-bit containers and complete32-code groups | Keep old rule addressable; it has neither byte-tail nor independent segment nor big-order fields | This is a distinct supported storage contract, not an exact old-field binding |
| `quantization_cute_dsl_utils.py:1755-1784,1931-1969` computes NVFP4 E4M3 scale from group max and coarse encoding multiplier g, then encodes E2M1 using the stored scale | `hierarchical_low_float_scaling@1` + the admitted E4M3/E2M1 group quantizer, with outer decode1/g | g is an encoding multiplier; do not apply it in the decoder's inverse direction. Strict RN and approximate reciprocal producer paths have separate finite-precision policies |
| `:1786-1895` constructs and scores both map4/map6 representations, chooses map4 only when err4<err6 | `quantization_candidate_error_selection@1(metric=absolute/squared,score_arithmetic=fp32_decoded/fp16_scaled_proxy,group_shape=vector16,outer_scale_max=256/448)` | Winning packed codes and stored E4M3 scale move together; ties choose map6. No new candidate is needed for this selector |
| `:1541-1609,1719-1752` strict score reconstructs stored E2M1/E4M3 with explicit FP32 products and either1/g or row_amax/(6*outer_scale_max) | Existing `pre_candidates`, `pre_select` require a coherent coarse-scale generation and full16-value score | Row-amax normalization is not interchangeable with an unrelated externally supplied global scale. The strict operation order remains part of the binding |
| `:1615-1716` proxy score decodes E2M1*E4M3 through an FP16 pair product, converts back to FP32, then accumulates FP32 differences/errors against x*g | Existing `score_arithmetic=fp16_scaled_proxy` | The FP16 product checkpoint distinguishes this score from strict decoded-domain scoring; it does not imply every error operation is FP16 |
| `quantization/nvfp4_quantization_utils.py:83-113,341-384` keeps outer maximum256/448, metric and proxy settings in the resolved recipe; scale builder uses that same maximum | Bind `outer_scale_max` and caller/producer/decoder recipe together | Code-generation fast math and error-proxy arithmetic are separate flags, not one combined accuracy switch. Zero tensor uses a separate scale-builder branch |
| `quantization_cute_dsl_utils.py:591-655` selects linear,128x4 or8x4 physical scale coordinates | Existing independent store/permutation refinements; this choice does not select new numerical scales |8x4 offset=32*((row//8)*ceil(padded_cols/4)+col//4)+4*(row%8)+col%4. Each32-byte tile covers8*4 distinct coordinates;128x4 uses the previously checked512-byte map |
| `activation.cuh:69-104` split CTAs traverse disjoint vector indices and stage unrolled inputs in T before converting | Local elementwise fusion plus vectorized transfers; final cast follows Activation(FP32x)*FP32y | Split changes ownership/transport and register lifetime. It does not introduce an intermediate activation cast before multiplication |
| `norm/kernels/rmsnorm_sigmoid_gate.py:152-169` normalizes each loaded output row then applies FP32 sigmoid gate before BF16 store | `operator_fusion@2(consumer_kind=row_reduce_then_map,intermediate_site=registers,fused_ops=norm_scale,checkpoint=none)` and admitted approximate sigmoid/rsqrt binding | Gate is after the statistic: normalizing x*sigmoid(gate) would change the row maximum/second moment and the requested operation |

The [logical check](checks/check_segmented_bit_packing.json) covers2048 cases:
both orders, all bit patterns at lengths0..8, combinations of three short/empty
segments, and lengths around the2048-value CTA loop boundary. For lengths3 and5,
big-order independent bytes are160 and104, whereas flat concatenation is one
byte173. Reading packed byte padding as semantic mask values is excluded.

The packing candidate restricts native prefix preparation to bounded Int32
intermediates and at least one segment; it does not claim a supported zero-CTA
launch. Individual zero-length segments are included. Pure boolean conversion
happens before this candidate, so truthiness of other input dtypes is not silently
part of its lossless claim.

Recipe composition: compute one coherent coarse NVFP4 scale -> construct both
complete code/local-scale candidates -> score the declared16-value domain ->
emit one paired representation -> independently permute local scale storage ->
consume with the matching coarse inverse. Repacking scales preserves that chosen
candidate; it cannot substitute the floor-anchor MX recipe or another error metric.
Native converter/reciprocal ranges and additional adaptive norm, rotary/cache and
model-specific layouts remain in the [auxiliary inventory](quantization-norm-inventory.md).
