# Single-position decode with transposed contractions

For a single query position, tile query heads that share one KV head along the
small MMA dimension and use KV tokens along the large dimension. The source
rewrites both contractions, rather than merely permuting a shape descriptor.
Sources below are under flash_attn/cute/ in the [pinned tree](inputs/sources.json).

Representative: SM100,FP16,head group8 sharing oneKV head,head/value dimension128,
token tile128,single query position. interface.py:1301–1321 selects SwapAB when
the resolved config requests it; other feature/shape restrictions are in config
and wrapper validation, not implied by this representative.

| Source choice | Rule/field or gap | Concrete effect / premise |
|---|---|---|
| flash_fwd_decode_sm100.py:1–11,348–397: St=K*Q^T and Ot=V^T*Pt | [transpose_attention_chain@1](delta/transpose_attention_chain@1/template.yaml) | St axes(token,head), normalize over tokens for each head, then output axes(value,head). Not tensor_axis_permutation@1's rank-three view-only domain. |
| :249–257,354–362: head tile1/2/4/8/16/32, token tile128/256, MMA first dimension128 | output_tiling@2 plus backend compute binding | Small head dimension pads to the MMA minimum. Valid-head count and MMA N extent differ. The32-head specialization redistributes per-head stats across lanes(:721 onward); its fragment mapping requires a separate binding. |
| :660–670,366–373: separate QK/PV warps, two alternating softmax WGs, correction WG | Multi-role specialization gap; physical_storage@1 for independent S/P/O |16warps total. Four TMEM S slots,four SMEM P slots,two TMEM O slots,two row-sum mailboxes; KV ring gets remaining shared capacity. This is not the regular SM100 forward role allocation. |
| :289–304,397–425: derive KV stages from memory, require K/Vt stage-byte equality, alias final O staging with KV ring | physical_storage@1 occupancy/capacity; explicit layout binding | O staging may overwrite the ring only after the last K/V reader. Shared transaction count requires equal physical K/Vt stage bytes, even when logical dimensions differ. |
| :1400–1535: alternate softmax streams feed parity O and denominator accumulators; correction uses two-step max history | online_rescaled_reduction@2 invariant; two-stream role/phase realization gap | Saved max[j] and max[j+2] correct an old parity slot; final merge includes a factor for the lagging slot. Both slots must reach one common frame before addition. |
| :1435–1438,1467–1470,1574–1613: guard consecutive-empty maxima and empty split stats | empty_support_normalization@1 subject to zero/finite premises |Empty/empty correction is1 instead of exp2(-inf-(-inf)); empty splits emit0 and-inf LSE. Different safe-reference implementations require the same admitted logical contract. |
| :1617–1622: unmatched tail arrivals are allowed because CTA exits after one tile | Persistent-reuse gap / explicit scope guard |A persistent version must balance/reinitialize the tail protocol. One-shot exit is part of this realization's scope. |
| :1670–1715 wrapper and file header: split output with natural LSE, shared caller combine | split_combine_kernel@1 rescaled_merge; exact split state binding pending combined trace |Unused per-sequence splits can exit without writing; consumers must read only valid splits. Empty written splits differ from unwritten out-of-count slots. |

Algebraic witness: transposed score columns(0,ln2,ln4) and(ln2,0,ln4), V=(1,3,5)
give outputs27/7 and25/7 after token normalization. Row-normalizing that matrix
would normalize over query heads and change the operator. With parity outputs
in references r0 andr1, finalr1 requires O=exp(r0-r1)*O0+O1 and the same
denominator correction, not O0+O1.

The core mathematical design is accounted for. Callback/window, paging/varlen
and scheduler restrictions remain with their shared feature groups; native
32-head redistribution, barrier tails and split numeric evidence remain open.
The source cites a CUTLASS example; it was not fetched and that paused library
study was not resumed.
