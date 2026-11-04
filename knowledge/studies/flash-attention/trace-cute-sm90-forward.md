# CuTe Hopper forward: masked state and stage ownership

When QK/softmax/PV consume a descending key-tile stream, track the normalization
frame separately from async instruction completion. A correct arithmetic recurrence
can still be broken by rescaling the wrong generation of O or releasing a shared
operand before its last MMA reader. Inputs and revision: [manifest](inputs/sources.json).
Paths below are relative to that source tree. This is not a new CUTLASS source study.

Representative: CuTe DSL on SM90a, FP16/BF16 operands and FP32 accumulators,
head_dim=head_dim_v128, batch1, one Q/KV head, Q length129, K length193,
causal end-aligned mask, no custom score/mask modifier, no paging/split/sink.
`flash_attn/cute/interface.py:1206–1230` selects the SM90 class with two KV stages.
`config.py:250–267` selects128×128, RS PV, intra-warpgroup overlap for this head
size. These are source heuristics and compiled bindings, not measured optima.

| Source choice / binding | Rule and field, or explicit gap | Evidence / constraint |
|---|---|---|
| `flash_fwd_sm90.py:100–126,215–235`: CTA tile128Q rows, warpgroup tile64rows; two MMA warpgroups plus one producer warpgroup | `output_tiling@2`: CTA128×128 and WG64×128; `warp_specialization@1`: producer_warps4, issuing_threads32, consumer_groups2, epilogue consumers | 384threads total; producer group contains128threads but only32 issue KV TMA. Q/O TMA and packed-head non-TMA branches have separate participation. |
| `config.py:460–476`, `flash_fwd_sm90.py:615,640`: split register budget240 per consumer,24 producer | `register_rebalancing@1` exact source binding | Non-TMA producer variant changes to224/40 for two consumer groups. This is a distinct capacity binding, not a source speed result. |
| `flash_fwd_sm90.py:100–126`: QK SS, PV RS by default; optional PV SS stores P in smem | `compute_instruction_selection@1`: wgmma_m64nNk16_f16, f32, ss/rs; `operand_precision_conversion@1` for FP32 P to operand dtype | The backend helper creates actual atoms; exact emitted mapping/instructions remain at the external CuTe lowering boundary. Head-dimension tails are padded and predicated by source helpers. |
| `:238–249,569`: Q,K,V stage layout and Q/O storage alias | `physical_storage@1` shared occupancies; `shared_resource_binding@1` selected backend swizzle | Q's last async reader must retire before O uses the same storage. Concrete swizzle atom supplied by dependency; no claim from “banked” name alone. |
| `:856–960,1083–1133`: Q full/empty phase and two K/V slots, matching producer/consumer sequence | `per_slot_handshake@1`, `multi_buffer_rotation@2` S2, `async_transfer@1` TMA binding | Every role uses the same tile/stage sequence, phase and tail; slot reuse waits for its last reader. KV producer acquires slots; consumers wait and release K/V independently. |
| `:1184–1284`, `block_info.py:35–68,129–164`: peel final length/causal/local boundaries, skip certified mask-free interior | `masked_tile_pruning@1`: causal, masked_first true, masked_steps2 for representative | This rule suffices for the dense causal shape within its menu. Custom callbacks still run on an otherwise interior block; arbitrary sparse-list selection is not explained by this binding. Saved CUTLASS mask draft is not imported. |
| `softmax.py:187–248`: running raw-logit maximum, four-lane row max, lane-local partial sums, exp2 with folded scale | `online_rescaled_reduction@2`: max_sum_acc,running_max,threshold0,gain0,descending,deferred_divide; `cross_lane_reduction@1`: max/add over4 lanes | Final sum reduction is deferred to finalize(:263); per-lane L cannot be interpreted as a complete row denominator beforehand. Positive final support and common-frame/precision premises apply. |
| `softmax.py:219–231,253–289`, `flash_fwd_sm90.py:1308–1312`: safe reference for masked rows, reset no-work sparse tiles, zero output and -inf LSE | [empty_support_normalization@1](delta/empty_support_normalization@1/template.yaml) | Store max=-inf as metadata, use safe finite0 for exponent subtraction. Empty numerator must be initialized0. NaN repair branch is not proof of empty support. |
| `flash_fwd_sm90.py:1355–1427,1500–1579`: prologue first QK/softmax, steady QK(next)+PV(previous), drain last PV | `cross_iteration_software_pipelining@1`: skew1,softmax_with_pv,async_wait_depth1 | wait_group1 retires QK before softmax reads S; wait_group0 retires previous PV before releasing its V and overwriting P. K release follows its QK completion. Prologue/epilogue make each PV execute once. |
| `:1521–1523,1568–1571,1414–1416`: move O correction earlier for value head>128 | `online_rescaled_reduction@2`: rescale_point before_next_mma versus per_tile; existing pipelining and lifetime obligations | Large-value branch saves the scale for the preceding P and applies it before its PV, including the drain. Representative128 takes later correction after previous PV completion. Correction must apply only to old-frame contributions. |
| `:301–309,712–759`: logical-length bounded V TMA descriptor for ragged/dense seqused layouts | `rebind_transfer_descriptors@2`: descriptor_count1,update_site direct_global, batch key; new scoped implementation binding, native publication not verified | Per-CTA128-byte slot; changed batch recomputes row end, updates extent, fences and prefetches descriptor. Masking P alone does not neutralize a NaN V padding value because0*NaN is NaN. Paged and stride0 broadcast V are excluded; old readers must retire before descriptor mutation. |
| `:340–351`: SingleTile, SingleTileLPT or SingleTileVarlen scheduler | Work assignment binding; **scheduler-specific mapping remains to compare** | Representative causal uses LPT mapping, not automatically the old FA3 atomic-counter persistent scheduler. Same while-shaped executor API does not establish persistent task reuse. |

## Boundary witness

Causal support is k<=q+(193-129)=q+64, not k<=q. CTA0 owns q0..127 and
visits key blocks1 then0. In its first block, q0 has no allowed keys; q127
allows128..191, while k192 is length-valid but causal-masked. In the next block,
q0 admits0..64, so its empty prefix becomes positive. CTA1 contains only valid
q128; it admits first-block keys128..192 (65 values), then all128 keys from
block0. Its other127 physical query rows are output tails. This separates length
masking, end-aligned causal masking, empty prefixes and physical output bounds.

For a frame witness, old state at r0 has L1,O3; the next logit is ln2 with value5.
At r'=ln2 the correction is.5, so L'=1.5,O'=6.5. Scaling after incorrectly adding
the new-frame contribution would instead give(3+5)*.5=4. In the overlapped
schedule, previous PV must finish before its O is corrected; moving that correction
earlier uses a saved scale whose generation matches the P about to be consumed.
The last-half drain must apply the final saved scale too.

Related SM90 local, paged, packed-GQA and sparse branches are inventory items
until their additional mappings/constraints are read. Hardware fragment layout,
TMA descriptor proxy semantics and native exp/convert errors remain implementation
obligations. No GPU compilation or throughput measurement was performed.
