# Attention: streamed state, selected support and recomputation

When scores cannot be stored as a full query/key matrix, distinguish three
choices: carry normalized-reduction state, choose the logical support, and
reconstruct probabilities later from a saved row statistic. Source paths below
are relative to `examples/` in the pinned retained tree from [manifest](inputs/sources.json).
The [starting pool](inputs/pool-snapshot.json) supplies the named versions.

| Source and concrete binding | Rule/choice | Boundary or useful difference |
|---|---|---|
| `flash_attention/example_mha_fwd_bshd.py:23–92`: FP16 Q/K/V, FP32 row max/denominator/output, Q tile reused through KV loop | `online_rescaled_reduction@1`, state `max_sum_acc`, per-tile correction, deferred divide; `operator_fusion@1`, softmax then matmul | Probability cast to FP16 precedes PV. Fast exp2 and cast errors belong to the numerical policy. Causal loop pruning still needs element masking at its boundary. |
| `flash_attention/example_mha_fwd_varlen.py` and GQA relatives | Segmented sequence origins and head grouping compose with tiling/online reduction | Query head maps to its KV group; sequence tails and causal positions use local sequence coordinates. Ragged packing changes addresses, not the normalization domain. |
| `flash_decoding/example_gqa_decode.py:50–188`: head tile, KV splits, normalized partial output plus base2 LSE stored as FP16 | `reduction_partition@1` then `split_combine_kernel@1`, `combine_op=rescaled_merge`, `partial_dtype=f16` | Merge uses weights `2**(LSE_s-LSE_all)`, not an arithmetic average of partial outputs. Source floor-sized splits need a covering, nonempty binding; arbitrary lengths/empty masked splits are not established. |
| `attention_sink/example_mha_sink_fwd_bhsd.py:49–138`: shifted causal/window positions, one sink logit per head, zero sink value | Available `online_rescaled_reduction@2`, running max, gain 1, denominator-only extra term in the same frame | Sink contributes `exp(sink-r)` to denominator only; it is not multiplied by attention's QK scale. Source adds it after the stream. Exponent range and masked-prefix handling remain premises; do not infer the source uses sink-inclusive running maxima. |
| `deepseek_mla/example_mla_decode_paged.py:16–129`: score = Q-content·KV + Q-pos·K-pos; same KV content supplies V | Two contractions share one score tile; `indexed_gather@1` page addressing plus online state and rescaled split merge | Page size must be a multiple of tile size. Intended physical row is `table[logical//page_size]*page_size + logical%page_size`. Source split path uses `(k*block_N)%block_size` instead of `(start+k*block_N)%block_size`; bind page-aligned split starts or retain this address mismatch. No bug investigation is needed for extraction. |
| `deepseek_mla/example_mla_decode_persistent.py:36–134`: static SM-sized worker grid, grid-stride split tasks, `T.sync_grid`, then merge tasks | `persistent_tile_scheduler@1` reuses work mapping; rescaled merge algebra reuses `split_combine_kernel@1` | Same-kernel cooperative grid barrier is a distinct synchronization binding, not the old second-kernel realization. All workers must reach it, partials must be visible, and launch/lowering must guarantee progress. |
| `flash_attention_sm100/mha_fwd_bshd.py:22–169`: `ss` stages P in shared; `ts` stages P in TMEM; both load score/output TMEM to registers | `physical_storage@1`, `staged_relayout@1`, instruction selection | `accumulator_operand_reuse@1` alone does not describe this transformed P: exp/cast is performed in registers and explicitly copied back. TMEM operand shape/layout and completion are native bindings. |
| Same SM100 file `flashattn_wasp:177–363`: softmax threads 0–127, DMA 128–159, MMA 160–191, two KV stages by default | `warp_specialization@1`, `multi_buffer_rotation@2`, per-slot handshake plus storage lifetimes | Softmax waits previous PV completion before correcting O. S/P/O TMEM are single allocations even though barrier arrays are staged: their actual reuse dependencies must be preserved. Native execution is not inferred from the source's fallback menu. |
| `flash_attention/example_mha_bwd_bshd.py:89–239`: save base2 LSE; preprocess Delta=sum(O*dO); recompute P per KV-owner tile | New `softmax_probability_recomputation@1`; contraction tiling; atomic partial reduction | dV=P^T*dO, dS=P*(dO*V^T-Delta)*scale, dK=dS^T*Q. KV owner writes dK/dV; dQ receives atomic partials into initialized FP32 storage, then converts to output type. Recompute/cast policy must agree with forward. Sink backward additionally uses `-P_sink*Delta`, then reduces query/batch contributions for each shared head sink (`attention_sink/example_mha_sink_bwd_bhsd.py:321–344`). |
| `flash_attention/example_gqa_bwd_tma_reduce.py:171–266` | GQA adds output reduction across query heads sharing KV; source offers TMA atomic-add or per-group scratch plus sum | `reduction_partition@1` and storage choice account for the ownership difference. TMA reduction needs target support/completion; it is not an ordinary non-atomic copy. SM100 backward relatives change native MMA/storage, not this derivative identity. |

## Selected support is part of the operator

`blocksparse_attention/example_tilelang_block_sparse_attn.py:35–126` and
`seer_attention/block_sparse_attn_tilelang.py:35–129` gate KV copies and both GEMMs
with a block mask. `block_causal_attention/block_causal_attention.py:19–34`
uses logical block coordinates: noisy/noisy diagonal admits equal blocks;
noisy-to-clean boundary admits earlier blocks; clean boundary includes equality.
These are concrete mask/traversal bindings. Seed `masked_tile_pruning@1` captures
the causal idea but does not provide every runtime support generator. Do not use
the paused CUTLASS mask draft as an admitted dependency.

`deepseek_nsa/example_tilelang_nsa_fwd.py:18–118` gathers selected blocks per
query/KV head (`indexed_gather@1`, row/block indices), masks their causal edge,
and shares each KV load across grouped query heads. Its reference is the current
tile maximum, not the running maximum: frame correction is algebraically valid
with finite references, but is outside version 2's two explicit reference policies
and can enlarge intermediate exponents. Keep that binding gap rather than claim
an immediate-max match.

`deepseek_v32/sparse_mla_fwd.py:15–166` instead gathers selected individual rows,
uses a running max and two content/position contractions, and restricts top-k to
multiples of `block_I=64`; dimensions are powers of two and large head groups
split in units of 64. Indices must be in bounds **before** the load, not merely
masked afterward. Duplicate selected rows have multiplicity unless their producer
deduplicates them. The paged sparse decode maps selected logical tile → page and
in-page tile (`blocksparse_attention/example_tilelang_sparse_gqa_decode_paged.py:80–129`);
first-tile-only tail masking assumes reverse-sorted indices. Empty split mass
requires a neutral LSE representation in addition to zero partial output.

The indexer is a separate computation: `deepseek_v32/fp8_lighting_indexer.py:90–172`
forms FP8 QK, positive scale, ReLU, weighted head reduction, then masks the valid
range. `dsa_hisa/hisa.py:30–130` mean-pools and requantizes K, selects coarse
blocks, scores only their raw tokens, selects tokens and restores sequence-local
indices. Boundary blocks are forced into coarse selection; BF16 coarse-score
selection can change ties/order. This is a hierarchical approximation, not a
proof that coarse losers cannot contain the exact global top-k. Fine-tune
`indexer_topk_reducesum.py` fuses score generation with incremental bitonic
top-k (the bounded `sorted_run_topk@1` gives a related sorted-run invariant,
but its k/run limits and deterministic ties do not directly admit this source); the radix selector is considered with the
library's other selection kernels.

`minference/example_vertical_slash_sparse_attn.py:75–310` combines contiguous
slash blocks (TMA two-stage producer/consumer) and gathered vertical columns
(cp.async prefetch) into the **same** online row state. Its index producer must
make that support union disjoint or explicitly preserve intended multiplicity.
`indexed_gather@1`, rings/handshakes and online reduction explain the transport
composition; there is no automatic dense-equivalence or top-k quality guarantee.

## Output-column sharding with shared normalization

`deepseek_v32/sparse_mla_fwd_pipelined.py:119–192` assigns score/softmax and left
output columns to WG0, right output columns to WG1. WG0 publishes P and the
correction factor each step; WG1 corrects exactly its old output before its PV.
Both release KV after use; the shared P/correction slot cannot be overwritten
until WG1 has consumed it. Both output halves divide by the same final mass.

`sparse_mla_fwd_seesaw.py:104–159,232–410` additionally alternates score tiles:
WG0 scores even tiles, WG1 odd tiles. They exchange maxima, choose a common
reference for each pair, correct old outputs and their separate denominator
partials, then exchange P. Each WG performs both PVs for its own output columns.
Final denominator is the sum of the two partial masses in the common frame.
Thus score work partitions along KV while output ownership partitions along D;
neither output half may consume only its own score tiles. This composes
`reduction_partition@1`, online state, warp roles and shared storage lifetimes.
The source aliases K-position storage with P after QK completion and aliases Q
with final output after all score uses. These aliases require last-use ordering.
The source's finite mask sentinel is a numerical binding: zero masked mass is
not an exact-real consequence of a large negative finite value. Paired-tile
tail participation, frame agreement and asynchronous last-use remain named
obligations; no new scheduling Template is needed just to rename this composition.

No GPU compilation, timing or end-to-end replay was performed in this source
survey. The new probability-recomputation pair records its algebra and checkpoint
premises; native layout, math/cast and launch obligations stay with each binding.
