# Absorbed MLA forward designs

MLA here computes softmax(Q*K^T + Qv*V^T)*V, with optional QK/rope term.
The latent V operand participates in both score construction and the weighted
output. Reusing its loaded bytes changes lifetimes and issue order; it does not
permit normalizing incomplete score partials. Sources start at flash_attn/cute/
in the [retained tree](inputs/sources.json).

| Design / source | Rule or explicit gap | Source binding and constraint |
|---|---|---|
| flash_fwd_mla_sm100.py:208–274: twoCTA absorbed path, value dimension split in two | reduction_partition@1 over score feature components; output_tiling@2 over value output halves | QK term and both QvV latent halves accumulate into the same score before softmax. O value halves have separate accumulators but share the same normalization. Head packing must satisfy the source cluster/tile divisibility or use the separate padded sparse path. |
| flash_fwd_mla_sm100.py:249–274: independent Q/K/Qv/V/S/P/O pipeline counts | physical_storage@1 plus per-slot concepts; twoCTA phase/native mapping gap | Q1,K1,Qv2,V4,S2,P1,Oi1,stats2,bitmask2 in the shown constructor. TMEM columns per CTA account for cluster group size, not merely the full logical output width. |
| flash_fwd_mla_1cta_sm100.py:1–33,2557–2616: M64 weight-stationary MMA, Q/rope in TMEM, latent Qv stationary | compute_instruction_selection@1 gives only tcgen05 family; Layout-E weight-stationary realization gap | A64×N accumulator occupies physical128 lanes×N/2 columns. QK zero-initializes S first; absent QK, first QvV split initializes it. Each later split accumulates, never zeroes prior score terms. |
| flash_fwd_mla_1cta_sm100.py:2578–2655: V loaded once, read under K-major score and MN-major output descriptor views | physical_storage@1 retained occupancy; byte-identical alternate-view mapping remains a realization obligation | Score consumer waits V but does not release it; PV consumes the same bytes through sVt and releases only after its async readers complete. Reloading V is not required by this source instance. |
| flash_fwd_mla_1cta_sm100.py:368–395: V capacity determines legal S-ahead schedule | Explicit resource/issue-order coupling | 16-bit path has two V stages for one block's two latent halves, so next S cannot precede previous PV release. FP8 has four stages for two blocks and admits S-ahead when requested. Source assertion requires stages >= two*latent_splits; recorded source performance comments are not study measurements. |
| flash_fwd_mla_1cta_sm100.py:120–145: sparse gathered token list, <=64 real heads padded in-kernel | grouped_head_row_packing@1 only for complete logical groups; guarded padding implementation separate | MQA-only,cp.async loads,no SplitKV,topk length positive multiple128. Single token's list serves all heads. Dense varlen/paged branches have different scheduler and tail constraints. |
| flash_fwd_mla_1cta_kb64_sm100.py:1–43,1749–1816:64-key blocks, three whole-latent stages, streaming per-part load readiness | reduction_partition@1 plus physical_storage@1; multi-frontier native publication gap | Each stage is64×512 latent values and four column-block parts. S begins on ready latent parts; rope arrives last. This is more than prefix_ready_contraction@1's two-region implementation and needs a distinct per-part binding. |
| flash_fwd_mla_1cta_kb64_sm100.py:1982–2053: dual packed Q halves produce two partial scores per key | reduction_partition@1,group_count2,combine add; shared-memory exchange/lifetime binding | Partner thread is tidx xor64. Partial scores are exchanged and added before mask, row max and exp. Normalizing each partial separately then combining probabilities would change the operator. Pair barrier is between the corresponding two warps, not a whole-CTA requirement. |
| flash_fwd_mla_1cta_kb64_sm100.py:14–22, mainloop: next S before prior PV | Existing software-pipeline invariant, tcgen05 realization gap | Q in TMEM frees shared capacity for a third whole-latent stage. PV reads the latent stage through MN-major view and only then releases it. S/P, O and latent-ring generation lifetimes must agree. |
| flash_fwd_mla_1cta_kb64_sm100.py:2218–2260: stream normalized O and rounding residual in32-column chunks | physical_storage@1 register occupancy; rounding_residual_checkpoint@1 | O_hi=cast(O_fp32), O_lo=cast(O_fp32-float(O_hi)), computed from the same normalized values. This is not carry_compensated_parameter_split@1's exact bit codec. It reduces register live range versus retaining a whole value split. |
| flash_bwd_preprocess.py:465–493; interface.py:871,1678: optional residual consumed in dPsum | rounding_residual_checkpoint@1 then softmax_gradient_row_checkpoint@1; [backward trace](trace-cute-backward.md) | dPsum uses dO*(O_hi+O_lo), reconstructing in FP32 before multiplication. Pairing, rounding policy and derivative error remain explicit obligations; a low-precision pair is not automatically exact FP32 reconstruction. |

Representative substitution: value dimension512 gives two256-feature QvV
score reductions and two256-feature O outputs. Score terms are added first,
then one softmax. If one token has partial logits(0,ln4) and another(ln2,0),
the summed logits are(ln4,ln2), so their probability ratio is2. Separately
normalizing the two components and multiplying/adding probabilities without
the corresponding common normalization does not supply that softmax.

For a loaded latent block, lifetime extends from its load through both score
and PV readers. The 16-bit1CTA two-half ring contains only one block; trying
S(next) before PV(current) has released those halves leaves no free latent
storage. FP8 four halves or the kb64 three-whole-block organization changes
that capacity dependency. Native scheduling benefit is not inferred from
the mere existence of a second implementation.

Paged cp.async, gathered validity bits, real-head guards and packed varlen
token mapping reuse the [indexing](trace-scheduler-and-indexing.md) and
[sparse](trace-sparsity.md) dispositions. Detailed native Layout E mappings,
per-part barriers, component descales, residual precision and store lifetime
remain named obligations. The twoCTA, oneCTA128-key and oneCTA64-key designs
are accounted for without claiming identical rounding or benchmarking them.
