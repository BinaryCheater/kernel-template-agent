# Quantized prefill: decoder placement and scale boundaries

Source paths below are relative to the retained revision in the
[source manifest](inputs/sources.json). Representative: classic FA2 prefill,
finite FP8 or NVFP4 E2M1 codes, FP16/BF16 compute operands. The new choice is
[decoded_tile_materialization@1](delta/decoded_tile_materialization@1/template.yaml),
with its [proof outline](delta/decoded_tile_materialization@1/proof.yaml).

| Source decision and binding | Rule field or gap | Evidence / boundary |
|---|---|---|
| `attention/prefill.cuh:156-176,403-445`: supported byte-sized KV, Dvo != 64, Dvo <= 256, Q tile > 16; FP4 also bounds Dqk and requires a consumer capability | `decode_site`, `pre_binding`, `c_capacity` | Repack selection is derived from shape, consumer implementation and shared-memory budget. Three FA2 launchers enable FP4; POD/persistent traits do not thereby acquire a repack implementation. Host/device budget selection must agree. |
| `:1436-1461`: cast sixteen FP8 codes into two sixteen-byte 16-bit output blocks | `pre_decoder`, `pre_layout` | Original per-token/head scales are outside this cast boundary. Shared layout uses a new row stride and the matching swizzle. |
| `:1483-1539`: eight payload bytes plus eight padding bytes represent sixteen E2M1 codes; one E4M3 scale per sixteen codes | `pre_decoder`, `decoder_and_metadata` | Decode two eight-code words, preserve scale conversion and 16-bit multiplication order, and write two output blocks. `vec_dtypes.cuh:483` selects the FP16/BF16 conversion helper. |
| `:3687,3765,4717-4845`: K decode, publication, QK, then V decode and PV | `pre_lifetime` | Shared decoded scratch may change roles only after all old readers retire. Ordinary 16-bit consumer loads must read the decoded coordinates. |
| `:1048-1170`: token/head scale staging follows contiguous or paged KV coordinates with independent strides | `pre_inputs`, indexed_gather@1 binding | Page table identity and code/scale generations must match. Inactive scale rows use 1, rather than introducing zero times an exceptional code. |
| `:1585-1775,4115-4145`: apply token/head K scale after the feature dot; apply V scale to V fragments | Explicit scale-site binding | K scale is constant across the dot's features. Group16 FP4 scales are not. Moving a V scale onto low-precision P can underflow; it is not implied by real-number distributivity. |

`fused_dequantization@1` supplies inline integer-weight decoding and its limited
group menu; it has no whole decoded tile choice or group16 E2M1 binding.
`physical_storage@1` handles lifetime/capacity of values already present, not
whether a decoder is repeated or materialized. Neither supplies the new choice.

The [logical check](checks/check_repack_and_pool.json) covers all sixteen E2M1
codes across adjacent unequal scale groups, 64 unique logical coordinates and
FP16 byte equality between the two modeled paths. It exposes a wrong-group
mapping. With P=0.5, V=256 and scale=2^-26, scaling V first gives 2^-19 while
scaling FP16 P first gives zero. These witnesses fix metadata and rounding
premises; native conversion instructions, BF16 rounding, shared swizzles and
publication still need device evidence. The source's software/hardware E4M3
scale conversion differs for NaN encodings; this candidate admits finite scales.

FP8 probability gain is a separate boundary: classic prefill scales P by 256
without scaling row mass, then compensates output by 1/256. Hopper uses the
format maximum instead. `online_rescaled_reduction@2` scales numerator and
mass together, so its common-gain field cannot express this numerator-only
quantization choice. The follow-up
[numerator_weight_gain@1 binding](trace-hopper-worklists-and-gain.md) now captures
that separate choice without changing the common-gain rule.
