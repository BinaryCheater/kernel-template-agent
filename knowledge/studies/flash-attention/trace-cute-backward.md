# Backward checkpoints and gradient ownership

When backward needs normalized probabilities and centered score gradients, two
independent choices matter: how P is retained/recomputed, and how its derivative
row center is obtained. A rounded forward output is a numerical input to the
second choice. Paths below are relative to flash_attn/cute/ in the
[retained source](inputs/sources.json).

## Representative and reusable choices

Take a finite positive-mass attention row with no dropout. The five backward
contractions are S=Q*K^T, dP=dO*V^T, dV=P^T*dO, dQ=dZ*K and dK=dZ^T*Q,
where dZ=P*(dP-center) is with respect to the scaled/modified score. The actual
score scale and callback chain rule remain attached to their derivative stage.

| Source choice | Rule / field or gap | Concrete binding and boundary |
|---|---|---|
| flash_bwd_preprocess.py:1-14,440-520: precompute dot(O,dO), optionally subtract dLSE | [softmax_gradient_row_checkpoint@1](delta/softmax_gradient_row_checkpoint@1/template.yaml), lse_gradient absent/present | Save one FP32 center per original query row. dLSE is the incoming natural-log normalizer gradient. Mask padded head dimensions with zero and guard row stores; unpacked varlen allocations cannot write slack into the next sequence. |
| flash_bwd_preprocess.py:535-549 and flash_bwd_sm90.py:1620-1645: convert LSE then reconstruct tile P | softmax_probability_recomputation@1, finite-row/base2 binding | Natural LSE is multiplied by log2(e), then P=exp2(score*scale_log2-LSElog2). Source replaces -inf LSE by zero for masked empty rows: this is a separate sentinel branch, not covered by the old finite-LSE rule. Exact excluded logits must remain -inf/zero probability. |
| flash_bwd_sm90.py:1620-1670: dP and row-center tile operation | New row-center rule composed with probability recomputation | QK and dO*V^T may overlap; wait for dP before P*(dP-center). The FP32 P used in the pointwise derivative is distinct from its FP16/BF16 MMA operand conversion. |
| flash_bwd_sm90.py:1669-1750: P/dS conversions and five-GEMM issue order | compute_instruction_selection@1 wgmma, physical_storage@1; per_slot_handshake@1 for data epochs | P can feed register/shared dV, while dS is staged for dQ/dK. With one P/dS stage, prior readers retire before overwrite. Q releases after dK; dO after dV. Named barriers join the actual participating WGs and the dedicated dQ store warp. Native lowering remains an implementation obligation. |
| flash_bwd_preprocess.py:465-493; flash_fwd_mla_1cta_kb64_sm100.py:2218-2260 | [rounding_residual_checkpoint@1](delta/rounding_residual_checkpoint@1/template.yaml), rounded_and_residual | Same normalized FP32 producer value yields BF16 high and BF16 rounded difference. Backward adds both in FP32 before dot(dO,O). Stream a row slice and retain its partial, rather than keeping three complete operand tiles live. |
| flash_bwd_postprocess.py:48-58,109-134,533-578: denominator-only sink derivative | softmax_gradient_row_checkpoint@1 denominator_sink finite_zero_value; reduction_partition@1 for already-defined row terms | Per-row -exp(sink-LSE)*center reduces to one shared sink parameter. One CTA per head owns the result, either dedicated or fused into selected postprocess work. The source's -inf-LSE sentinel giving sink_prob1 needs its own semantic binding; finite-normalizer rule does not infer it. |
| flash_bwd_postprocess.py:689-739: accumulator output conversion | Existing map/scale and physical layout binding | FP32 dQ scratch is read in the producer's actual tiled layout, multiplied by the remaining score scale and cast. This scale must be applied once, not also blindly inserted into an already-scaled derivative operand. |

For P=(1/4,3/4), V=(2,6), dO=2, the output is5 and center10. The
score derivative is(-1.5,1.5). Incoming dLSE=0.5 changes the center to9.5
and derivatives to(-1.375,1.875), whose sum is0.5. With ordinary
probabilities(1/4,1/2), sink probability1/4 and the same values, center is7;
ordinary derivatives(-0.75,2.5) plus sink derivative-1.75 sum to zero.
The [logical check](checks/check_backward_checkpoints.py) compares all four
ordinary/sink and absent/present-dLSE cases against finite differences. It also
shows that BF16 high+low is neither a lossless FP32 codec nor correctly decoded
by BF16 addition. [Observed results](checks/check_backward_checkpoints.json)
are host arithmetic checks, not native kernel or gradient certification.

## Output ownership and architecture differences

| Source / design | Disposition | Consequential choice |
|---|---|---|
| flash_bwd.py:882-1064,1071-1197: SM80 register-MMA backward | Same probability/center algebra; compute_instruction_selection@1 mma; reduction_partition@1 defines contributors | A KV-owned CTA retains dK/dV across query blocks, while dQ receives cross-KV FP32 atomic adds. For GQA, dK/dV also sum query-head contributors. P/dS staging and V-in-register storage change occupancy, not the derivative identity. Atomic completion/order needs a backend binding. |
| flash_bwd_sm120.py:1-62 | Same SM80 device body with narrower capacity gate | Dtype FP16/BF16, head dimensions divisible8, key tile divisible16, threads divisible32. Shared capacity is queried for SM120; V-in-register mode aliases Q/V storage via max rather than sum. New architecture name does not imply tcgen05 use. |
| flash_bwd_sm90.py:1818-1893,1904-2048: dKV and dQ accumulation | reduction_partition@1 for contributor sets; ordered_global_combine@1 is only a partial match | Source combines bulk FP32 adds, optionally semaphore-ordered by KV partition or grouped query head. The old rule's read/modify/write program, co-residency scope and by_k/arrival menu do not directly express every source schedule. Name this ordered-bulk-add realization/rank-map gap rather than claiming full reuse. |
| flash_bwd_sm90.py:1988-2048: ordering and skipped local work | Completion, rank and progress obligations | Before deterministic release, wait for global bulk adds to complete (read=False). Ordinary atomic mode may wait only for shared-source reads when reusing staging. Local-mask work skipped by a CTA still advances required dense order counters; otherwise a later contributor can wait forever. SPT uses reversed KV rank derived from the row's admissible key upper bound. |
| flash_bwd_sm100.py:48-76,3690-3952: TMEM alias versus dedicated P/dS slots | physical_storage@1 plus per_slot_handshake@1; schedule and cross-warp mapping binding | Hdim64 oneCTA has spare TMEM and gives P/dS dedicated slots, releasing S after loads so QK(next) overlaps current softmax. Aliased variants must join all S/dP readers before overwrite. Independent S pipeline phases cannot be derived from another channel's parity. At twoCTA a pre-commit barrier also orders every warp's dS exchange before the single DSMEM sender; removing it as in oneCTA is invalid. |
| flash_bwd_sm100.py:139-177: twoCTA pipelined/serial schedule and M64 layout | physical_storage@1, output_tiling@2; exact cluster layout/schedule binding | Head<=128 uses distinct Qt/dOt/exchange buffers and interleaves the next S with current derivative GEMMs. Head192/256 chooses serial issue and aliases Qt/Q, dOt/dO and, when present, exchange/dQ staging. Tile_n64 has half-row TMEM packing and stages P/dS in shared memory; its peer dS slice is already contiguous, so no extra exchange buffer is allocated. These layouts change which overwrite barriers and completion handshakes are necessary. |
| flash_bwd_sm100.py:4111-4167,4320-4468: deterministic tiled bulk reduction | Ordered-bulk-add/rank-map gap, with concrete protocol | Rank may use sparse write-order metadata and cluster-local subgroups, not raw n_block. Delayed semaphore release overlaps staging but cannot precede completion of that row's adds. sdQaccum alias with exchange storage requires every bulk reader drained before the peer overwrites it. No native progress or release/acquire proof claimed. |
| sm100_hd256_2cta_fmha_backward.py:28-106,185-231 | Distinct output-ownership decomposition; [device follow-up](trace-specialized-backward-and-combine.md) | Fixed head/value256, twoCTA instructions, no local/custom score/mask. It launches a query-owned dQ kernel then a KV-owned dKdV kernel with separate tiles128x128 and128x64. Direct dQ output avoids the ordinary cross-KV dQ scratch/reduction path by repeating score/probability work across owners. |

## Sparse MLA differences identified

flash_bwd_mla_sm100.py:220-255 fixes the V ring to the number of latent
splits (two): dP's stationary dO half is indexed by that ring stage. Changing
only V stage count changes operand identity. The dV accumulator ring is
different: head64 can use three stages, mapping (group,split) to
(2*group+split) modulo3 on both producer and consumer. Uniform
multi_buffer_rotation@2 fits the latter once this logical generation is bound;
it does not authorize inferring a feature split from arbitrary ring parity.

At :2906-2940, the saved-P mode reads unnormalized BF16 P, scales it in FP32
by exp2(scale*row_max-LSElog2), rounds once for dV, and retains FP32 normalized
P for dS. This is not softmax_probability_recomputation@1: its representation
and checkpoint error differ. The recompute mode at :2941-3058 uses LSE and
recomputed scores; it forces invalid gathered positions to -inf before exp and
zeroes invalid dP before multiplication to avoid zero*NaN. The bitmask-disabled
path requires a stronger caller validity premise.

At :3270-3320, dV fragments scatter through the original top-k indices with
0<=index<seqlen_k guards and FP32 vector atomics. Duplicate indices are
multiple gradient contributors and cannot be deduplicated silently. Existing
indexed_gather@1 is read-side only; scatter_reduce_output_shards@1 is disjoint
cooperative peer ownership, not a collision map. Indexed additive scatter is
covered by indexed_additive_scatter@1 in the
[specialized follow-up](trace-specialized-backward-and-combine.md), which also
resolves saved-weight decoding, chunk scratch and dQ/dQv/dK decompositions.
Native atomic and coordinate bindings remain explicit obligations.

## Choice review

The default is two independent candidates: checkpoint encoding and gradient
row contraction can vary separately. A single combined attention-specific rule
would hide that independence. Failure is observable through the retained
non-lossless/low-add witnesses and the derivative row-sum/finite-difference
checks. Neither candidate introduces an agent policy or a fixed precision
threshold. A new format or empty-row convention needs an explicit binding;
native rounding, synchronization and gradient-error checks stay obligations.
