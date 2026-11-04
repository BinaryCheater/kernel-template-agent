# Pair supply, QK normalization and cache destinations

When a feature transform needs the original opposite component, consider local
pair ownership, repeated partner loads or a converged warp exchange. The new
[paired_transform_supply@1](delta/paired_transform_supply@1/template.yaml) and
[outline](delta/paired_transform_supply@1/proof.yaml) express that implementation
choice while keeping the requested pairing, position coefficients and input
checkpoints fixed. Paths below are relative to the retained [tree](inputs/sources.json).

| Decision and concrete binding | Rule field / boundary | Evidence |
|---|---|---|
| `include/flashinfer/pos_enc.cuh:105-199`: adjacent pairs use local vector snapshots; split halves load the matching other half | `paired_transform_supply@1(pairing=adjacent,partner_supply=local_vector)` or `(pairing=split_halves,partner_supply=duplicate_load)` | Pairing is the consumer contract, not an interchangeable layout toggle. Prefix R is even; the suffix remains unchanged. Split-vector alignment must keep both half maps valid |
| `norm/fused_qk_rmsnorm_rope.cuh:496-511`: one head/warp, split-half partners arrive from lane XOR16 at the same fragment component | `partner_supply=warp_exchange`, `pre_exchange` | For head width D with D/32 components/lane, lane XOR16 maps feature i to(i+D/2)%D. A cross-warp partner does not inherit this binding |
| `pos_enc.cuh:692-808`: position-ID frequencies, optional smoothing, head-stride dispatch and two heads loaded before either store | `pre_coefficients`, `pre_layout`, `pre_alias` | Loading both heads before store is a transport schedule; it does not by itself prove every cross-thread in-place partner read completed. In-place duplicate loads require the native read-before-overwrite premise |
| `norm/fused_qk_rmsnorm_rope.cuh:381-459`: Q and K each reduce across all their own heads, weight, then RoPE; V only copies/converts | `operator_fusion@2` with complete statistic dependency and explicit checkpoint; generic reduction rules | Denominator is num_heads_this*head_dim, not head_dim. Q/K statistics are separate; V has no norm or rotation |
| `:445-459`: non-FP8 path rounds normalized/weighted values to BF16 and back before RoPE; FP8 specialization skips that cast | `pre_numeric`: distinguish the two numerical recipes | Source comment that coarser FP8 makes the checkpoint irrelevant is not a general equivalence proof. Logical witness below produces different FP8 codes |
| `:469-528`: 3D positions derived from token-in-sequence, frame/height/width patch geometry and feature bands; adjacent and split representations use their pair frequency index | `pre_coefficients`: pair map, position/frequency generation and target recipe explicit | A flat token ID or cache slot is not the same position. Band widths, sequence geometry, YaRN/attention gains and frequency precision remain source-specific contracts |
| `pos_enc.cuh:938-1094`: fused rotate/scale/append uses pos_ids for coefficients, batch_indices plus positions for cache destination | `paired_transform_supply@1` -> `operator_fusion@2(fused_ops=scale_and_cast)` -> indexed/page-mapped stores with unique destination premise | Q and KV encoding multipliers are separate. K rotated prefix and nonrotated suffix write their correct cache regions; GQA/MHA also append V, MLA writes KPE/CKV without that V branch |
| `:945-959`: negative batch index skips token; otherwise page_iter=floor((indptr[batch]*page_size+position)/page_size), entry=remainder | Exact semantic locator; physical page lookup remains cache layout | Skipped tokens also skip Q work in this representative. Duplicate destinations have no winner-election protocol here: require unique destination writes or externally established ownership |
| `include/flashinfer/concat_mla.cuh:141-205`: load one shared RoPE fragment, broadcast by reuse across a fixed head chunk, pipeline next no-RoPE head load before current stores | Vectorized transfers and cross-iteration load/use schedule; concatenation preserves semantic fields | Output is [per-head no-RoPE | shared per-token RoPE]. Head chunk/source strides and alignment are native bindings; no quantization is implied |
| `include/flashinfer/mla_kv_pack_fp8.cuh:89-161`: per warp/token, BF16 [K_nope128|V128] per head and shared PE64 convert to saturating E4M3 | Local scale-and-cast fusion with explicit fixed source/output field map | Key192=[K_nope128|PE64], value128=V. Lanes0..7 convert PE once; shuffles broadcast its8-value pieces across head groups. No dynamic group scale or footer is produced by this packer |

The [logical check](checks/check_paired_transform_supply.json) covers248 pair and
suffix cases plus448 XOR16 feature coordinates. It also shows why original-pair
capture matters: rotating(1,2) with c=0,s=1 should give(-2,1), but writing the first
component before rereading it gives(-2,-2).

For x=1.051025390625,y=0.125,c=0.6,s=0.8, BF16-rounding inputs before the FP32
rotation gives E4M3 output0.5625; retaining FP32 inputs gives0.5. This is a
logical counterexample to unconditional checkpoint removal. The source FP8
specialization remains an explicit alternative recipe, not evidence of native
bitwise equivalence. The check also distinguishes across-head RMS normalization
from independent head normalization.

Cache identity has three separate inputs: RoPE position, logical destination
position and physical page map. A witness with RoPE positions[11,7], cache
positions[4,0], batch indices[0,1] and nonadjacent physical pages resolves to
(page4,entry0) and(page17,entry0). It would be wrong to substitute cache positions
for the coefficient IDs. Publication to readers requires completion of the whole
payload generation; ordinary split K/V stores do not promise atomic records.
The earlier [complete-record election](trace-fp8-correction-and-cache-writes.md)
is a different source path with its own collision policy.

The [rotary/cache inventory](rotary-cache-inventory.md) retains the full family:
frequency variants and generated model routes are grouped by mechanism. Native
in-place/compiler ordering, special coefficients, packed cache consumers and
communication-overlap bodies remain named realization/design follow-ups.

The separate model append entry contracts are also distinct:
`flashinfer/cake_fused_qk_rope_append.py:16-40,117-120` uses per-head128-feature
weights and allows no norm, RoPE then norm, or norm then RoPE. Those stage orders
are requested semantics, not free fusion permutations. It derives a request-local
append position from q_indptr/seq_lens and looks up physical pages accordingly.
`cake_fused_qk_rope_fp8_append.py:16-30,131-148` further separates dynamic Q
per-token/head scales from a static Q encoding multiplier, while K/V parameters
are dequantization scales used as divisors. Their native generated bodies remain
an explicit follow-up; these entry contracts do not inherit the across-head3D
representative or pos_ids/positions map above.
