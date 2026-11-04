# Rotary, cache and pack design inventory

Source paths refer to the retained [tree](inputs/sources.json). The
[representative trace](trace-rotary-norm-and-cache.md) separates pair supply,
normalization scope, numeric checkpoints and physical cache destinations.

| Design family | Source routes | Current disposition |
|---|---|---|
| Direct/ragged/position-ID rotation | rope.py -> jit/rope.py -> csrc/rope.cu -> pos_enc.cuh | Adjacent local pairs, split-half partner loads, active prefix/identity suffix and head-stride/two-head scheduling traced. Native in-place duplicate-load ordering remains an obligation. |
| Cached coefficients and long-context frequencies | rope.py:apply_rope*_cos_sin_cache/apply_llama31*, pos_enc.cuh, cutile/rope_quantize_fp8_cutile.py | Table coefficients and computed/smoothed frequencies are distinct input contracts under the pair rule; interleave table index is pair index. Exact backend coefficient math, native cache chunk maps and extreme-position error remain named. |
| Across-head QK norm +3D RoPE | norm/fused_qk_rmsnorm_rope.cuh | Full Q or K head axis statistic, V copy path, three position bands, XOR16 supply and BF16 checkpoint/FP8 alternative traced. Host target menus, YaRN coefficients and boundary geometry remain native bindings. |
| Per-head/model fused QK append | cake_fused_qk_rope_append.py, cake_fused_qk_rope_fp8_append.py, cake_fused_qk_rope*_backend, jit/cake_*rope* | Relate to local fusion and pair rule only after binding per-head statistic, scale and cast stages. Entry contracts distinguish no norm/RoPE-then-norm/norm-then-RoPE, per-head128 statistics, request-derived positions and dynamic/static Q versus K/V decode scales. Generated native representatives and output/collision contracts remain. |
| Quantized rotate/append | rope.py:rope_quantize_fp8_append_paged_kv_cache -> pos_enc.cuh, page.py/page.cuh | Distinct RoPE IDs, logical positions and physical pages traced; MLA KPE/CKV versus GQA/MHA K/V roles and unique-write premise explicit. Ordinary append is not complete-record winner election. |
| Standalone KV codec/materialization | quantization/fp4_quantization.py:nvfp4_*kv*, kernels/nvfp4_kv_materialize.py and jit/cache paths | Existing decode/scale/generation rules and prior mixed-cache traces are candidates for reuse. Concrete standalone compressed/uncompressed endpoint map and scale direction remain. |
| MLA concatenation/context pack | concat_ops.py -> concat_mla.cuh; mla_kv_pack.py -> mla_kv_pack_fp8.cuh | Shared PE across heads, head-local no-RoPE/V split, load-ahead and fixed saturating E4M3 pack traced. Additional source stride/type menus remain realization bindings. |
| QKV exchange/overlap | jit/cake_qkv_norm_rope_exchange_overlap.py and generated csrc package | Couple normalize/rotate chunks with distributed exchange; retain for communication-family tracing, not ordinary elementwise fusion closure. |
| Sparse/model cache insert | mla/_sparse_mla_sm120/_cake_dsv4_nvfp4_rope_insert.py, jit/cake_dsv4_nvfp4_rope_insert.py | Relate to earlier SM120 record/election and decoder traces; model-specific compressed-prefix/rope suffix ownership remains a reconciliation item. |

Advance by these distinct differences; no per-generated-entry audit is required.
Sampling, recurrent, MoE and distributed families remain in the full-library
[inventory](inventory.md).
