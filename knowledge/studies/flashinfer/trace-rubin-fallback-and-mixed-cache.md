# Rubin fallback identity and mixed compressed-cache consumers

Sources are relative to the retained [tree](inputs/sources.json). This pass
keeps computation identities separate from native pipeline and approximation
bindings; the candidate count stays ten.

## Preferred/fallback cluster work is one logical domain

`flashinfer/cute_dsl/attention/rubin_mtp/kernel.py:6836-6851` prepares the
Blackwell two-CTA fallback by re-viewing output/workspace rows as two row
blocks. It shares the preferred four-CTA scheduler parameters and accumulator
views; it does not create a second independent schedule of the same outputs.

At :7681-7704 and repeated role loops such as :7908-7934, a scheduler tile
(raw_m,raw_y,raw_split) maps to native fallback coordinates:

- pair rank = raw_m % 2;
- native query block = 2*decoded_query + raw_m//2;
- original decoded batch and split identity are retained.

The four raw M coordinates therefore become two native row blocks with two
cooperating CTA ranks each. The preferred M256 view folds two H128 query
tokens; causal masks must still decode the original H128 token identity.
This is a concrete bijection premise for the mixed-cluster graph gap in the
[earlier trace](trace-mla-cute-cutile-cache.md). It does not establish the
entire mixed launch/scheduler contract from coordinate arithmetic alone.

CLC fetches use a one-stage, 16-byte response pipeline (:6858 onward). Every
role consumes the same work response and advances after its last required use;
selected early-compute roles can prefetch the next response while finishing
current work. Bind per_slot_handshake@1 to that response channel only with the
actual consumer count and native completion protocol. persistent_tile_scheduler@1
has no preferred/fallback cluster decoder or CLC work-stealing field, so this
remains a mixed execution/scheduler extension rather than a changed binding of
its static-stride/atomic-counter menu.

The [logical check](checks/check_mixed_cache_channels.json) substitutes two
folded query blocks and four raw M coordinates, retaining split7. The eight
native (rank,query,split) coordinates cover query0..3 and pair ranks0/1 once.
Native preferred/fallback execution, remote P/correction publication, resource
branch allocation and CLC cancellation/ownership remain realization obligations.

## FP8 and FP4 cache sections share one canonical consumer

Source directory:
`include/flashinfer/attention/sparse_mla_sm120/kernels/dsv41_fp8/`.
`decode_schedule.cuh:10-107` separates gather and conversion groups. Each
section describes its format, row/page geometry, valid index and source base.
Footer pages load payload and scale portions separately; inline records load
one complete record using the validated row pitch. Both invalid payload and
metadata select the immutable zero row.

| Source decision | Reuse / explicit gap | Concrete binding |
|---|---|---|
| `decode_schedule.cuh:30-80`: gather raw FP4 when section.extra, otherwise canonical FP8 | indexed_gather@1 plus existing boundary/zero-row binding | Format controls copied bytes and readiness channel. Footer selected_data_offset/selected_scale_offset select matching data/scale geometry together. A slot's capacity and current-format record width are separate from page pitch. |
| :83-107: conversion waits raw-ready only for FP4, advances SelectiveSlotPhases only there, then publishes KvReady for every tile | empty_work_pipeline_projection@1 per-channel generation premise, per_slot_handshake@1 | Raw-ready is conditional; KvReady is unconditional. RawFree release remains per tile, with format-dependent acquisition of KvFree on the appropriate producer/converter role. Skipping raw-ready does not mean skipping every control event. |
| `convert.cuh:39-91`: two group16 E2M1/E4M3-scale groups decode to FP32, regroup into32 values, choose one UE8M0 scale, round to E4M3 | Decode + regroup + requantization chain; extra numeric-policy binding | Canonical FP8 stage is not identical to the original decoded FP4 values. The source scale is 2**ceil(log2(max(amax/448,1e-4))) and uses rounded division before the power boundary decision. Its native casts and error policy are not supplied by decoded_tile_materialization@1's equal-decoder premise. |
| `convert.cuh:94-115`: converted FP8 row bytes and scale bytes use canonical consumer layout | physical_storage@1 + explicit code/metadata mapping | Conversion output must publish both data and its new scale before math reads. Raw storage becomes free only after conversion; canonical KV storage remains live through all math readers. |

With two slots and formats FP4,FP8,FP8,FP4,FP4,FP8,FP4,FP4, slot0 raw events
occur at tiles0,4,6. Their raw phases are0,1,0, whereas global tile-ring phase
would give0,0,1. The logical check distinguishes that wrong phase sequence.
The ordinary KV-ready channel instead has four generations per slot. This is
concrete reuse of channel projection, not a new barrier rule per format.

The same check gives a rounding witness: one group16 scale is 2^-9 and the
adjacent group's scale is448. E2M1 code0.5 gives a nonzero value2^-10 in the
small group; code6 in the large group makes amax2688. The regrouped UE8M0 scale
is8, so the small value normalizes to2^-13, below half the minimum E4M3
subnormal. Nearest-even E4M3 conversion gives zero. This refutes a lossless
reformat explanation; the native converter's range/rounding and workload error
remain separately admitted implementation evidence.

## NVFP4 PV changes the quantization axis

`kernels/dsv4_nvfp4/decode.cuh:300-334` writes BF16 P in a common block frame,
then all256 math threads reuse the current QK source tile to prepare one
short-lived transposed V stage while producers gather the next source tile.
`compute/nvfp4_vt.cuh:53-108` transposes 16x16 E2M1 codes across lanes,
reads the **original token's scale** for each decoded value, then requantizes
the transposed group. The source token-major scale group and the destination
feature-major key group are different quantization domains. Transposing only
packed codes and keeping original scales would mix unrelated token metadata.
This is decode -> axis permutation -> group quantization, not a byte-only
transpose or decoded_tile_materialization@1's fixed-decoder equivalence.

`kernels/dsv4_nvfp4/decode.cuh:348-362` retains a separate BF16 rope value
contraction. :380-457 applies direct-output sink mass once and uses declared
LSE output units. Existing output tiling, online-state and empty/sink rules
provide the algebra only after that split contribution, numerical gain and
state representation are bound. `kernels/dsv32_fp8/decode.cuh:154-233` uses a
separate 512/576-feature raw layout and query staging; do not infer its source
record ABI from the DSV4.1 no-rope or NVFP4 layout names.

The compressed consumers now have concrete format/conversion/conditional-channel
dispositions. Remaining extraction gaps include axis-requantization policy,
mixed-cluster runtime ownership and family-specific prefill/decode tile maps.
Native transpose lane maps, FP4 MMA scale placement, zero-row publication and
pipeline arrival counts are implementation obligations. Continue the remaining
SM120/target variants and sparse/training families, using these shared choices
rather than adding one record for each model name.
