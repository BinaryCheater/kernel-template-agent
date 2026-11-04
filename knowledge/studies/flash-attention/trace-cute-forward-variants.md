# Forward architecture and storage variants

These are differences from the [SM90](trace-cute-sm90-forward.md) and
[SM100](trace-cute-sm100-forward.md) representatives, not separate source audits
of every specialization. Source paths below start at `flash_attn/cute/` in the
[pinned tree](inputs/sources.json).

| Distinct source choice | Rule/field or gap | Binding and boundary |
|---|---|---|
| `flash_fwd.py:579–618,1145–1214`: SM80 synchronous warp MMA, cp.async loads and register P | `compute_instruction_selection@1`: mma_sync_m16n8k16_f16,rr,f32; `async_transfer@1`; `operator_fusion@2` softmax_then_matmul | 128threads in interface, one KV stage by default; QK consumes staged Q/K then softmax corrects O before PV. Unlike Hopper WGMMA, there is no evidence of an in-flight tensor instruction while CUDA executes softmax; load overlap remains possible. |
| `flash_fwd.py:604–625`: optional Q register retention permits Q/V shared-buffer reuse | `physical_storage@1`: smem shared occupancies and register Q occupancy | With Q_in_regs false, Q and V storage sum; with true, shared Q/V capacity is max(Q,V), with additional live registers. Q must be copied out before its shared bytes are overwritten. This is a storage/lifetime choice, not a new attention algorithm. |
| `flash_fwd_sm120.py:15–67`: subclasses SM80 path, forces internal arch branch80, checks actual SM120 capacity | Same compute/storage bindings, new target/capacity instance | Can-implement permits f16/bf16, dimensions divisible by8,tile_n divisible by16,thread count multiple32 and row/thread divisibility. Actual DSL target remains SM120; internal Arch.sm80 selects code paths. Do not infer TMEM/tcgen05 support from the Blackwell name. |
| `flash_fwd_sm100.py:203–218,128–134`: two-CTA MMA spans two adjacent Q row blocks | `output_tiling@2` custom cluster→CTA mapping; **cooperative two-CTA MMA realization gap** | For work_m2,q_stage2,group2, stage0 CTA0/1 own Q blocks8/9; stage1 owns10/11. Each CTA still owns128rows; a twoCTA MMA has256rows/stage. This is not two independent CTA outputs with duplicated inputs. |
| `flash_fwd_sm100.py:958–990,1113–1188`: cluster pipeline counts and coordinated TMEM allocation/deallocation | Per-slot ownership concept, **cluster publication/count/retirement binding remains open** | Non-MMA participant counts multiply by group_size2; transaction bytes also multiply by2(:589). Initialization waits at cluster scope. Non-TMA Q rejects group2(:3478). The one-CTA prefix_ready_contraction@1 is not silently widened to this case. |
| `flash_fwd_sm100.py:344–352,2116–2166,2912–2926`: q_stage1 S ping-pong alternates S/P slots but one O accumulator | Physical storage/multi-buffer concepts with explicit generation binding; **tcgen05 software-pipeline realization gap** | S slot=count&1, phase=(count>>1)&1; q_stage1 and supported head/tile menu required. QK(i+1) precedes PV(i), so S(i+1) completion no longer proves O(i) complete. Correction explicitly waits O_full before any rescale/read. Regular two-Q-stage schedule uses a different program-order implication. |
| `flash_fwd_sm100.py:546–562,3553–3565`, `config.py:870–887`: three KV slots pack K192/V128 with phase-shifted middle address | [alternating_size_ring_packing@1](delta/alternating_size_ring_packing@1/template.yaml) | OneCTA only,kv_stage3. L=tile_n*192*element_bytes,S=tile_n*128*element_bytes. Slot1 shifts by±(L-S)/2 around stride(L+S)/2; producer converts its phase with xor1, consumer uses direct phase. Native padded footprints and cross-slot last-reader ordering remain unverified. |
| `flash_fwd_sm100.py:729–730`: K and V reuse one physical allocation, with different logical shapes | physical_storage@1 plus packing candidate; not uniform multi_buffer_rotation@2 alone | At tile_n128,FP16, packed bytes131072 versus uniform147456. Capacity bounds alone do not prove overlap safety. The [logical geometry check](checks/check_alternating_storage.json) separates frame geometry from transition lifetime. |
| `flash_fwd_sm100.py:222–226` and epilogue helpers: SplitKV value-dim256 emits FP32 partial output in128-column chunks | `physical_storage@1`: smaller epilogue staging plus sequential chunk binding; split-combine state is separate | Source uses chunked_split_epi only exact dim256 and disallows TMA O in this mode. Two128-column chunks cover the256 output columns; scratch can be reused after each store's readers retire. This preserves KV capacity at the cost of multiple epilogue chunks. |
| `softmax.py:421–461`, `utils.py:826–845`: hybrid SFU and polynomial exp2 | approximate_math_rewrite@1 only describes the function rewrite; **per-fragment backend mixture/range/mask error gap** | 32-scalar fragments, pairs selected by frequency/residue, protected first fragments and always-SFU last fragment. Polynomial clamps inputs below-127, extracts integer exponent and applies degree3 fractional polynomial. This shifts SFU work to packed arithmetic, with no new speed claim. |
| `softmax.py:438–458`, `utils.py:826–845`: mask sentinels enter hybrid exp2 | **Strict zero-mass invariant needs a separate numerical binding** | Polynomial emulation of -inf follows clamp to-127 rather than mathematically exact zero. The narrow P cast may yield0 while the pre-cast FP32 denominator sees a tiny positive term. Thus masked_terms_are_zero in online_rescaled_reduction@2 and exact L=0 detection in the empty-support rule are established only for an appropriate zero-preserving backend/binding, not automatically for all hybrid configurations. This is an abstraction boundary, not a bug investigation. |
| `flash_fwd_sm100.py:254–265,2687–2691`: SM103 row maximum during TMEM load | **Fused load/max realization gap** | Hardware max is used only with no callbacks and no per-tile mask; masked tiles still reduce post-mask values in software. An unmasked hardware maximum can be a safe scaling reference in some formulations, but it is not the selected rule's semantic masked maximum without an admitted alternative policy. |
| `flash_fwd_sm100.py:623–631,1502–1530`: per-CTA bounded ragged V descriptor, delayed acquire | `rebind_transfer_descriptors@2`: direct_global,one descriptor,batch key; scoped backend obligation | Rebind only on batch change; defer acquire until the first V load so independent K/Q issue can precede it. Bounds apply to logical V rows; format, private slot lifetime and fences must match the current transfer. Native helper lowering remains external. |

## Storage geometry and a required lifetime edge

For normalized footprints L12,S8, the shifted frames are:

| Logical wrap parity | Slot0 | Slot1 | Slot2 | Capacity |
|---|---|---|---|---|
| 0 | [0,12) large | [12,20) small | [20,32) large | 32 |
| 1 | [0,8) small | [8,20) large | [20,28) small | 32 |

Each frame is disjoint, but a newly large slot0 overlaps the preceding phase's
middle large operand on[8,12). Waiting only for slot0's previous small operand
is insufficient if that middle operand still has a native reader. Source
interleaves/releases PV and next QK across Q stages(:2020–2055); the exact last
SMEM read relative to release/commit is a native binding obligation. The candidate
therefore requires retirement of **all overlapping occupancies** and does not
assert that per-slot source barriers alone prove it. The logical check passed
L/S cases12/8,49152/32768 and16/16; it does not emulate GPU instruction timing.

For S ping-pong, count0/1/2/3 maps to slots0/1/0/1 and phases0/0/1/1. If the
new QK completes before the old PV, waiting only for the S event permits correction
to race the old O writer. The source's explicit O_full wait closes exactly that
missing edge. Native event count and TMEM-fence correctness still require their
own implementation evidence.

The dense SM80/SM120 differences and the SM100 storage/transport variants above
are now accounted for by reuse or named gaps. Decode/SwapAB, sparse/MLA, detailed
scheduler mapping, backward and auxiliary CUDA groups remain open in the library
inventory. No end-to-end replay or new GPU measurement was performed.
