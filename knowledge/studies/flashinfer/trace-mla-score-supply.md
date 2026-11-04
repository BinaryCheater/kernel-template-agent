# Classic MLA: two feature owners, duplicated or exchanged scores

Route: `flashinfer/mla/_batch_mla/_backends/fa2_backend.py:20-125` delegates to
`_fa_common.py:40-68`; `flashinfer/jit/attention/modules.py:255-305` builds the
FA2 module and `csrc/batch_mla_run.cu:17` includes `attention/mla.cuh`.
All source paths refer to the retained [revision](inputs/sources.json).
Representative: FP16/BF16 Q/KV, 512 latent features, 64 rope features,
64 packed query rows, two 128-thread groups in one 256-thread CTA.

For query/head row i and token j the score is
Qlatent_i dot C_j + Qrope_i dot Krope_j. Both components accumulate before
mask/max/exp, and C_j also serves as the value. Absorption is an input contract;
this kernel does not derive the absorbed query from original projection weights.
The output is the latent-space weighted sum, not a silently reconstructed
per-head value projection.

| Source choice / binding | Rule or field | Evidence and applicability |
|---|---|---|
| `attention/mla.cuh:77-133,1243-1277`: 256 threads, query tile 64, output latent split into two halves | output_tiling@2; [score_replication_or_exchange@1](delta/score_replication_or_exchange@1/template.yaml), `c_shape` | Output feature ownership stays fixed while score supply changes. With Dlatent=512 each group owns 256 output features. This is not a transposition of the attention chain. |
| `:584-614`: rope contribution initializes scores when present, latent contribution accumulates; otherwise latent initializes | reduction_partition@1 over feature components | First initialize, then accumulate all score components. A softmax of each component is a different computation. |
| `:401-429,454-485,497-578`: QK_SHARD true computes separate key halves and exchanges maxima; false duplicates all scores/normalization | New candidate `score_supply` / `pre_online` | Sharded owners agree on the row frame before exponentiating and correcting old output/mass. Replicas each hold a full normalization state. |
| `:615-713`: cast exponent weights to FP16/BF16, use those cast weights for row sums and PV; sharded branch publishes P and both groups read all keys | `pre_numeric`, `pre_exchange`, `pre_output` | This source's denominator uses rounded P, unlike the high-precision pre-cast mass in numerator_weight_gain@1. P covers the entire key range for both value halves. |
| `:714-747`: sum two mass shards only for QK_SHARD | `mass_multiplicity`, `pre_replicas` | Each replica already has complete mass. Adding replica masses would halve O. Shard masses share each tile's frame before final combination. |
| `:51-74`: rope/P share per-stage storage; max/mass tables are a union; all live tile storage aliases final O staging | physical_storage@1, candidate `pre_backend` | One two-owner FP32 row table needs 8*query_rows bytes, reused between maxima and final masses. P needs 2*query_rows*key_tile bytes per current exchanged generation; raw/decoded operands and ring stages are additional allocation. Max readers finish before mass overwrite, rope readers before P overwrite, latent readers before final O overwrite. |
| `:1216-1277`: shared-memory regimes derive (stages,key tile,QK_SHARD)=(2,64,true),(2,32,true),(1,16,false); source thresholds 221696/147968/92672 bytes per SM | Candidate `pre_backend`, derived source binding | These are dispatch menus, not independent runtime knobs or per-block capacity proofs. Actual allocation/launch must fit the selected device and cooperative grid. |
| `:310-385,1260-1264`: E4M3 KV forces key tile 32 and QK_SHARD false, casts/decodes KV before 16-bit compute | decoded_tile_materialization@1 local-decoder binding | Latent scale may be per tensor or per physical token/128-feature group; the scale index follows the same page map as the codes. Rope scale remains a separate binding. Reuse decoded latent values through both score and PV readers. |
| `:978-1055,1205-1213`: per-plan work ranges, then cooperative grid join and in-kernel split-state merge | Existing online/split-state algebra; cooperative merge realization gap | Grouped_head_row_packing@1 maps packed rows back through num_heads before causal masks/output stores. All CTAs must reach the grid join, and launch residency must support it. A separate combine-kernel rule does not itself implement this cooperative phase transition. |

The new [proof outline](delta/score_replication_or_exchange@1/proof.yaml) retains
the score, mass multiplicity, full-key and alias premises. Existing
attention_value_axis_partition@1 fixes four owners and changes key-partial/full
value output ownership to feature ownership; it does not choose redundant
complete score production. softmax_probability_recomputation@1 reconstructs
backward P from checkpoints, rather than computing duplicate forward scores.

The [logical witness](checks/check_score_supply.json) uses two four-key tiles,
eight identifiable value features and separate latent/rope score contributions.
It checks common-frame exchange, old-state correction, rounded-weight mass and
matching complete feature outputs. It distinguishes duplicate full denominators,
reading only a former key half, and separately normalized score components.
These checks do not execute native row-sum MMA, fragment mappings, barriers or
the cooperative grid merge. No performance claim follows from duplicated versus
shared scores without a workload/resource measurement.

## Remaining MLA relatives

FA3 builds `csrc/batch_mla_sm90_run.cu`, which includes `attention/mla_hopper.cuh`.
Its traits/storage at :68-173 and score/PV operations at :444-550 use WGMMA,
separate producer/consumer roles and FP8-to-BF16 staging. It therefore needs a
role/frontier binding rather than inheriting the classic group's protocol from
the shared KernelTraits name. The retained `mla_swa_sm90/` tree adds dedicated
mailbox/owner-PV pieces. CuTe DSL monolithic/modular/Rubin-MTP, cuTile, SM120
sparse MLA and Kimi/DSV4 cache variants also remain distinct inventory work.
CUTLASS and external TRTLLM backend internals are not inferred from their Python
capabilities; the paused CUTLASS study remains untouched.
