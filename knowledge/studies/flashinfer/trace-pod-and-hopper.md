# POD work sharing and Hopper attention differences

Source paths are relative to `include/flashinfer/` in the retained
[source revision](inputs/sources.json). This trace separates a new work-pool
choice from existing attention algebra and unresolved native bindings.

## Two independent families in one launch

`attention/pod.cuh:39-122` gives each CTA one prefill/decode preference from a
per-SM arrival counter. If Np <= Nd, the preference cycle has floor(Nd/Np)+1
tags and only tag zero prefers prefill; the converse gives decode the last tag.
A global family counter supplies a unique job ticket. An invalid preferred
ticket triggers one claim from the other family. This is one job per CTA,
not a persistent worker loop. `:123-170` maps tickets to request/query/chunk/head
coordinates; `batch_pod.cuh:59-146` uses the same claim principle with batched
prefill descriptors. These are instances of
[two_family_cta_pool@1](delta/two_family_cta_pool@1/template.yaml) and its
[proof](delta/two_family_cta_pool@1/proof.yaml).

| Source choice | Candidate field / obligation | Concrete binding |
|---|---|---|
| `pod.cuh:395-418` launches sum of family job counts | `c_grid`, `exactly_once_and_coverage` | Atomic tickets are unique; a CTA can fail both families only after every valid ticket has been claimed. Np+Nd CTAs suffice independently of preference. Claiming does not imply computation has completed. |
| Same launch uses maximum body threads and shared bytes | `c_resources`, `uniform_dispatch_and_barriers` | Representative has two 128-thread bodies. Individual early exits in unequal-thread bodies do not establish legal full-CTA barrier participation. |
| Leader publishes family/ticket in first two shared integers, two CTA joins precede body scratch reuse | `pre_atomic`, `pre_barriers` | All roles read the same dispatch and control readers finish before scratch is repurposed. |
| Static per-SM plus two global counters, memset for each call | `pre_state`, `counter_lifetime` | Invocation exclusivity and completion before reset are required. Source warns that nsmid need not equal active-SM count; allocated span must cover observed SM IDs. |
| `:442-462` merges each family's split outputs afterward | `pre_jobs`, `completion` | Existing split_combine_kernel@1/state algebra applies with separate support and completion bindings. |

`persistent_tile_scheduler@1` supplies homogeneous repeated tile fetches, not
this heterogeneous one-shot family selector and maximum resource envelope.
The [logical check](checks/check_repack_and_pool.json) enumerates 900 serialized
preference sequences for family sizes 1..4, confirms unique complete coverage,
and separates undersized grids and unreset counters. The proof handles abstract
atomic tickets; the check does not enumerate asynchronous atomic interleavings.
SM-ID span, concurrent invocation exclusion and native body participation remain
binding obligations. Ratio preferences make no guaranteed runtime-share claim.

## Hopper relatives and retained gaps

| Source representative | Reuse or distinct difference | Disposition |
|---|---|---|
| `attention/hopper/kernel_traits.cuh:30-115`, `mainloop_mma.cuh:30-91,175-225,317-326` | Q tile multiple of 64, head features multiple of 32; one producer warpgroup plus compute warpgroups; SS QK, RS PV; reverse KV traversal; overlap next QK with previous PV | Reuse existing online-state and producer/consumer pipeline rules with explicit wait/retire bindings. Last V release after final PV is required for producer progress. V-transposed ordinary-16-bit layout is a view, not a physical transpose. V/O alias needs disjoint lifetimes. |
| `hopper/attention_updater.cuh:172-243` | Base2 state, finite empty-row reference, guarded zero row mass, correction of old O before new terms | Concrete online_rescaled_reduction@2 binding with gain 1 and explicit empty-state representation. No general exceptional-value repair follows. |
| `hopper/quantization/kernel_traits.cuh:93-153`, `mainloop_load.cuh:249-298` | FP8 64x64 V transpose uses paired byte/register permutations; original V released after transpose commits; Vt retained through PV | Same paired operand/output permutation gap recorded in FlashAttention FA3. Keep original-V and Vt generations distinct; do not invent a native permutation witness from layout names. |
| `hopper/variants.cuh:129-166`, `quantization/mainloop_mma.cuh:120-130,264-276` | After row-mass update, P is multiplied by format maximum (448 E4M3 / 57344 E5M2), cast and permuted; final factor is Vscale/Pgain | Numerator-only probability gain remains a numerical-policy gap. ODequantize is empty because finalization already includes PV compensation; do not multiply Vscale twice. |
| `attention/scheduler.cuh:1040-1194`, `hopper/tile_scheduler.cuh:81-184` | Sort by descending KV length, reverse query tiles, greedily assign each task to least accumulated estimated-cost CTA; flatten per-CTA lists and offsets; each CTA walks its own range | New preplanned cost-partition/list scheduling gap. runtime_work_descriptors@1 has binary-search chunk sizing/padding and one unit per CTA; persistent_tile_scheduler@1 has static stride/atomic fetch. Neither exposes greedy cost assignment plus per-CTA list traversal. Exact cover is semantic; the cost formula is a placement heuristic. |

`hopper/prefill_sm90.cuh:443-497` chooses shared-head versus per-head schedules;
`:538-607` selects source shape/stage menus. These are dispatch/compiled menus,
not measured performance. Blackwell, SM110/SM120, sparse, MLA and experimental
relatives still need their own distinct-design dispositions. No CUTLASS
submodule is fetched and its paused study remains untouched.

The two abstract gaps above are now extracted as cost_partitioned_worklists@1
and numerator_weight_gain@1. See the [follow-up binding](trace-hopper-worklists-and-gain.md)
for field comparisons, source substitutions and logical witnesses. This does not
close the native layout/pipeline obligations or remaining architecture families.
