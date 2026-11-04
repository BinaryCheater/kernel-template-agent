# Hopper: cost-partitioned lists and numerator-only gain

Source: the retained [revision](inputs/sources.json), paths relative to
`include/flashinfer/`. Two independent objects are captured by
[cost_partitioned_worklists@1](delta/cost_partitioned_worklists@1/template.yaml)
and [numerator_weight_gain@1](delta/numerator_weight_gain@1/template.yaml), each
with a sibling proof outline. They resolve the abstract-choice gaps in the
[earlier Hopper trace](trace-pod-and-hopper.md); native pipeline/permutation
bindings remain separate.

## Immutable per-CTA work lists

| Source choice / binding | Candidate field | Disposition |
|---|---|---|
| `attention/scheduler.cuh:1053-1112`: order requests by descending KV length, visit query tiles in reverse, and visit heads outside that loop | `pre_tasks`, ordered task input | Ordering is a caller policy. Zero-query requests emit no tile. Task identity includes request, head, query tile, offsets and lengths; sorting never changes the original output coordinates. |
| `:981,993-1001,1113-1127`: cost is 2*CTA_Q+effective_KV; nonfinal causal tile ends clamp kv_len-qo_len+(tile+1)*CTA_Q to [0,kv_len]; final tile ends at kv_len | `pre_cost`, least-load assignment | CTA_Q=128 except Dvo=64 selects 192. Current FA3 path does not pack group heads into query tiles, so the causal cost uses group size 1. Cost is an estimate, not measured cycles. |
| `attention/heap.h:30-66`: pop minimum accumulated cost, append complete descriptor, add task cost and reinsert | Program / `partition_cover` | Heap compares only cost. The logical check uses worker ID to break ties and verifies coverage rather than claiming the same tied worker assignment as C++ standard-library heap operations. |
| `scheduler.cuh:1131-1194`: flatten seven parallel IdType arrays and emit worker prefix offsets | `pre_descriptors`, `pre_ranges`, storage | For 32-bit IDs this is 28 bytes per task plus 4*(workers+1), excluding allocator padding, reserve capacity and temporary host vectors/heap. Every array must use the same permutation. |
| `hopper/tile_scheduler.cuh:81-188`: CTA b starts at work_indptr[b], advances by one and stops before work_indptr[b+1] | `worker_iteration`, `range_and_lifetime` | Descriptor list length may vary per CTA. This is neither a grid stride nor a global atomic next-work counter. |
| `scheduler.cuh:1103-1110`, `hopper/tile_scheduler.cuh:198-340`, `hopper/prefill_sm90.cuh:443-497`: when max_num_works_per_head>4096, plan one head and restore head from blockIdx.y | `pre_replay` | Lists are shared across head grid slices. Stored head placeholder is not consumed as the actual head. Actual grid has workers*num_heads CTAs; metadata storage is still one copy. |

`runtime_work_descriptors@1` handles runtime chunk size/padding and descriptor
emission, but its traversal is one unit per CTA and its balancing menu lacks
least-load list assignment. `persistent_tile_scheduler@1` exposes static stride
or atomic fetch, not per-CTA prefix ranges. The new rule supplies this missing
assignment/traversal object without widening either older version.

The [logical check](checks/check_worklists_and_gain.json) instantiates CTA_Q=128,
two heads and requests (Q,KV)=(260,600),(64,100),(0,500),(129,80). Twelve tasks
cover 906 valid query/head rows once, retaining a zero-support causal prefix as
an output task. Three worker ranges preserve the complete descriptors; ten
workers exercise empty ranges. Replicating a single-head plan reconstructs the
same full task identities. Deliberately mixing one metadata array's original
order with the flattened order fails that identity check. Native publication,
all-role agreement and per-task pipeline/accumulator retirement still need a
selected device binding; task cover alone does not prove those events.

## Gain P without gaining row mass

| Source choice / binding | Candidate field | Disposition |
|---|---|---|
| `hopper/variants.cuh:129-166`: G is 448 for E4M3 or 57344 for E5M2; final PV scale is per-KV-head Vscale/G | `gain_policy=format_max`, `c_gain`, `pre_reference` | Vscale is constant over that head's key contraction. ODequantize is empty because finalization already applies it. |
| `hopper/quantization/mainloop_mma.cuh:120-130` and subsequent iterations: update row state, then PQuantize, cast and permute P | `pre_mass`, `pre_layout` | d is the original higher-precision mass, not sum(Q(G*w))/G. Backend permutation still needs its native coordinate binding. |
| `hopper/attention_updater.cuh:180-243`, `quantization/mainloop_mma.cuh:264-276`: correct old numerator, final PV, normalize with scale_pv/d, compute LSE from original reference+d | `gain_and_frame`, `empty_and_final` | Correct old terms before adding fresh gained terms. LSE has no log(G) adjustment. Zero mass uses the explicit nondividing output branch; empty LSE representation is a separate contract. |
| `attention/prefill.cuh:3952-3965,4231`: value-split FP8 P is gained by 256 and output is compensated by 1/256 | `gain_policy=power_of_two`, `gain_log2=8` | Another binding of the same numerator-only choice. This does not claim every classic path uses that policy. |

In one frame Q(G*w_j)=G*w_j+delta_j gives output equal to the original
weighted numerator/d plus sum(delta_j*V_j)/(G*d), before additional native
arithmetic error. Reference changes transport old residuals with the old
numerator. Gain must stay fixed over a live numerator; changing it requires
state conversion. This is an approximation contract, not bitwise equivalence
to ungained weights or a claim that every gain improves accuracy.

`online_rescaled_reduction@2` applies a common gain to mass and numerator, then
compensates LSE. Its gain cannot encode this split. `shifted_weight_checkpoint@1`
saves per-tile reference/weight state for later normalization and has no gained
PV accumulator or final inverse-gain choice. These comparisons preserve the
older meanings and add an independent narrow-operand refinement.

The logical check enumerates finite nonnegative E4M3 encodings to model
nearest-even conversion. A weight 2^-12 vanishes ungained but survives after
G=448. Two tiles move the reference from 0 to 1 and test transported rounding
residuals, original mass/LSE and single compensation. A weight 0.3 forces a
nonzero residual so quantized-mass renormalization differs. The
[initial witness correction](checks/worklists-gain-witness-correction.json)
records why all-exact initial weights failed to distinguish those contracts.
The check does not validate native FP8 casts, exponent/MMA implementations,
E5M2 rounding or workload error bounds.
