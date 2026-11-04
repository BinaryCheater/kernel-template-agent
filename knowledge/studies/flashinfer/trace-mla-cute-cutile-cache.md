# MLA CuTe/cuTile routes and SM120 cache representation boundaries

Paths are relative to the retained [source tree](inputs/sources.json). This pass
records concrete reuse and new boundaries; it does not close every remaining
MLA backend or execute a compiler.

## Route selection is not a new kernel mechanism

The batch-MLA adapters resolve implementations explicitly:

- `mla/_batch_mla/_backends/cute_dsl_monolithic_backend.py:59-60` ->
  `cute_dsl/attention/monolithic/mla_decode.py` and FP16/FP8 bodies.
- `cute_dsl_modular_backend.py:65-68,112-120` ->
  `cute_dsl/attention/wrappers/batch_mla.py`, importing the modular MLA bodies.
- `_cutile_prepared.py:181-184` ->
  `attention/kernels/cutile/fmha_decode_bsr_cutile.py`.
- `cute_dsl_rubin_mtp_backend.py:42-45` ->
  `cute_dsl/attention/rubin_mtp/mla_decode.py` and `kernel.py`.

These are separate selected bodies, even when they share a public plan/run
contract. Compiler capability checks are applicability gates; missing compiler
features do not authorize a fabricated implementation or an inferred fallback.

## Concrete shared choices

| Source decision | Existing rule / field | Source binding and remaining boundary |
|---|---|---|
| `cute_dsl/attention/monolithic/mla_helpers.py:338-374`: flatten query token/head as token*H+head | grouped_head_row_packing@1 coordinate invariant | H<=128, M=128. H64,Q3 gives 192 rows, two tiles and tail 64; the second tile begins at token2/head0. Variable-query scheduling must retain original request/token/head metadata, not treat each M tile as one query token. |
| `monolithic/mla_decode.py:111-185`: occupancy-derived split count, cap by available KV tiles and reducer capacity | reduction_partition@1 plus runtime_work_descriptors@1 only where emitted descriptors match | Split count is derived from launch/workload policy. Workspace and reducer capacity must use the same effective split count; a smaller static reducer specialization cannot consume arbitrary runtime splits. Descriptor traversal still requires its selected scheduler binding. |
| `monolithic/mla_decode_fp16.py:2380-2458`: first QK then loop QK(next)/PV(previous), final PV drain, Q release after its last score reader | cross_iteration_software_pipelining@1, per_slot_handshake@1, physical_storage@1 | S/P/O use separate pipelines and TMEM fragments. Actual tcgen05 completion, reference-correction ordering and storage views remain realization obligations, not implied by a generic pipeline name. |
| `attention/kernels/cutile/fmha_decode_bsr_cutile.py:980-1088`: choose normal or transposed QK, accumulate rope before softmax, transpose PV consistently | transpose_attention_chain@1 with already-complete latent+rope scores; reduction_partition@1 | Transposed score shape is [tokens,heads], softmax token axis is 0 and PV computes V^T*P, retaining [features,heads] until final transpose. The normal path uses token axis 1. Both rescale old O before fresh PV. |
| Same cuTile body :1000-1034 and :1064-1088 | online_rescaled_reduction@2 gain=1 with source-specific numerical domain | Tail uses finite -1e6 score fill, exp2 flush-to-zero and approximate division. Exact masked-zero behavior requires a score/range binding; it is not the unconditional -inf mask premise. Empty/final state needs the selected path's explicit branch. |
| Same file :96-158,165-208: split normalized output/base2 LSE, actual split count from sequence length, padded LSE axis | split_combine_kernel@1, empty_support_normalization@1 representation mapping | Invalid split rows are zero-loaded and masked, including non-power-of-two counts. The empty-row divisor guard prevents an unconditional zero division; numerical floor and reduction MMA are explicit source bindings. |
| Same file :224-272: multi-page gather via advanced-indexing TMA or balanced per-page load/concatenation | indexed_gather@1 `granularity=page`; physical transfer realization | Compiler feature chooses the implementation. Page IDs must map both latent and rope fields to the same token generation; no claim about transaction count follows from one high-level gather operation. |

cuTile MLA dispatch at :1575-1587 selects transposition for fewer than 64 query
group heads only when the target supports it. This is a derived dispatch
heuristic plus a legality gate; it is not measured performance or a new
transpose identity. The older generic BSR entry has a separate predicate and
must not be used as proof of MLA target legality.

## Rubin mixed-cluster boundary

`cute_dsl/attention/rubin_mtp/mla_decode.py:70-100` restricts the selected path
to FP8 E4M3FN input/output, H128, uniform Q2/Q4, latent512, rope64, page64/128,
no variable split-KV and no DCP. `:227-284` prepares a preferred four-CTA 2+2
kernel plus a two-CTA fallback; folded reducer rows pair two real query tokens
into M256 but causal identity remains H128 with fold ratio2.

`rubin_mtp/kernel.py:179-204` separates QK/softmax CTAs 0/1 from PV CTAs 2/3.
The QK pair publishes FP8 P and row correction metadata into the PV pair's
shared buffers over DSM. This avoids recomputing full QK for two latent halves.
The payload and its row frame must publish together, and the PV pair's remote
ready votes join before a two-CTA MMA can read both operand halves.

This is a **mixed cluster execution graph gap**: persistent_tile_scheduler@1
provides homogeneous static/atomic work fetch, warp_specialization@1 partitions
warps within a CTA, and row_owned_cluster_state_combine@1 merges completed raw
states. None selects runtime preferred/fallback cluster shape while preserving
one QK/PV work identity, folded causal rows and cross-pair transfer frontiers.
The source also requires CLC/mixed-cluster compiler features. Do not generalize
the four-CTA graph to the fallback from the class name alone. Follow its actual
branch and native ready/release operations before proposing a reusable rule.

## SM120 footer versus inline scale records

`include/flashinfer/attention/sparse_mla_sm120/model/dsv41_layout.cuh:6-46`
defines no-rope DSV4.1 geometry (512 latent features), E4M3 codes and UE8M0
scales per32 features. Footer and inline layouts share the logical decoder
but not the address formula. `model/kv_storage.cuh:52-107` gives:

| Layout | Payload byte offset | Scale byte offset |
|---|---|---|
| Footer, page-local row r | r*DataBytes | page_size*DataBytes+r*ScaleBytes |
| Inline, flat record r | r*(DataBytes+ScaleBytes) | r*(DataBytes+ScaleBytes)+DataBytes |

For 512 E4M3 bytes and 16 scale bytes, footer Page64 row3 has payload offset
1536 and scales at 32816 within that page; inline record3 has payload offset
1584 and scales at 2096. Page base and inline flat-record identity are different
caller contracts. Passing one address convention to the other decoder swaps
metadata even when both report 528 bytes per token.

`dsv41_layout.cuh:55-78` separately defines E2M1/group16 E4M3-scale records:
256 payload bytes plus32 scale bytes, yielding288-byte inline records. Those
scales are not the UE8M0/group32 decoder. Existing physical_storage@1 describes
capacity/lifetime and indexed_gather@1 describes row selection; neither supplies
this code/scale placement map. decoded_tile_materialization@1 can consume a
fixed decoder only after the actual code/metadata coordinates and arithmetic
are bound. A new storage-address binding, not a universal dequantization claim,
is the current disposition.

Continue from the model-layout consumers and cache writers, plus monolithic
FP8 correction/quantization and the Rubin fallback graph. The modular body's
resource/frontier differences remain open. No new candidate or redundant
logical replay is added solely for these backend names.
