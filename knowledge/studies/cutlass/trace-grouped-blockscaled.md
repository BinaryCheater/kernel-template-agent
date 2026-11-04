# Blackwell block-scaled grouped GEMM

This slice covers both entries in
`examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_grouped_gemm/`:
`grouped_blockscaled_gemm.py` (SM100) and
`sm103_grouped_blockscaled_gemm.py` (SM103). Paths and method lines are retained in
[variant bindings](inputs/grouped-blockscaled-variants.json). Source paths below
are relative to the pinned tree in inputs/sources.json. No compiler or device run
is part of this trace.

## Contract and distinguishing witness

The shared witness uses FP4 E2M1 A/B, E8M0 scales with vector32, FP16 C,
MMA128×128, cluster1×1 and L1. Its three (M,N,K) problems are
(128,128,768), (256,128,0), (128,256,1536). Both retained frontend predicates
accept this witness. SM100's public dtype gate accepts FP4/FP8 inputs, not FP6;
its UNPACK helper branches do not broaden that gate. Its public gate also rejects
FP4 output even though a zero-fill branch mentions it. SM103 is FP4-only with
N128/256 and FP32/FP16/BF16/FP8 output. See checks/variant_contracts.py.

M/N cluster counts are [1,2,2]; K0 does not remove its two output tiles. With two
persistent workers, worker0 visits linear indices0,2,4 and worker1 visits1,3.
The group/local pairs are respectively (0,0),(1,1),(2,1) and (1,0),(2,0).
This exposes three independent clocks: problem metadata switches, nonzero-K
input/accumulator work, and output-ring generations for every output tile.

## Source decisions and rule ownership

| Decision and source | Binding and classification | Rule field, evidence or gap |
|---|---|---|
| `python/CuTeDSL/cutlass/utils/grouped_gemm_persistent_tile_scheduler.py:232–265,327` creates persistent group state | Retained prefix cursor; `use_cached_problem_shapes=False` in the entry callers. The helper's True path is a separate exposed helper binding, not a kernel constructor knob. | [online_group_work_decode@1](delta/online_group_work_decode@1/template.yaml), variable cache_shapes. Both paths retain the monotone cursor. |
| Same scheduler:513–534 computes work counts;536–584 decodes coordinates | ceil(M/clusterM) × ceil(N/clusterN); no K factor, pinned L1. Group-local cluster index decodes M-major, then adds the local CTA offset. | New rule owns prefix inversion only. Spatial group-to-CTA lifting remains external glue; multi-batch work counting needs another implementation. |
| Same scheduler:585–714,715 onward locates group and loads shape | Warp scans32 groups using inclusive prefix ends and popcount(query>=end); invalid lanes count zero. A hit32 advances the window. Optional two per-lane shape windows preserve group identity. | New rule program/inverse_prefix/cache_identity. checks/grouped_work_descriptors.py compares a window model with independent bisect, including empty groups and invalid queries. Backwards queries give a retained counterexample. |
| SM100:1022–1047,1085–1166,1203 onward; SM103:885–931,993–1039 | Private global descriptor workspace per physical CTA. SM100 input publisher owns A/B/SFA/SFB tuple4; SM103 splits AB and SF into tuple2 each. Cache last consumed group and update only on key change. | [rebind_transfer_descriptors@1](delta/rebind_transfer_descriptors@1/template.yaml), descriptor_count4 or2. Same-key reuse needs immutable metadata and compatible fixed transfer fields. |
| SM100:1912–2048; SM103:1673–1769 constructs current tensors | Load group base pointers and MN/K strides; A=(M,K,1), B=(N,K,1), C=(M,N,1), batch stride0, assumed alignment16. SF views use tile_atom_to_shape_SF with the fixed vector length. | New descriptor rule coordinate_identity. Changing rank, dtype or fixed atom layout requires a separately supported binding/variant. Scalar checks track problem identities and logical output domains, not encoded descriptor bits. |
| `python/CuTeDSL/cutlass/tensor_utils/tensormap.py:118–190` | Selected SMEM-shadow mode initializes fixed atoms, updates shadows, commits/waits read groups, synchronizes a warp, then release-copies shadows into global slots; each issuing reader acquires descriptor pointers. Five128-byte slots total per CTA. | New rule descriptor_lifetime/publication. The direct-global helper branch is outside this binding. Exact retirement/proxy ordering for each input/output engine remains an implementation obligation; the scalar model does not establish it. |
| SM100:1360–1367 scheduler role publishes group metadata | Eight fields include valid/group/tile/shape/K-count. Seven fixed warps: epilogue0–3, MMA4, input TMA5, scheduler6. | Metadata pipeline binds synchronization roles; warp_specialization@1 cannot encode one-warp MMA with its four-warp consumer-group unit. Existing role-granularity gap remains. |
| SM100:1710–1816; SM103:1565–1647 | C descriptor publisher owns tuple1 and updates for every changed output group, including K0. A K0 tile bypasses accumulator wait/TMEM read, fills registers with zero, and still stores C. | New descriptor rule skip_key plus explicit grouped K0 output variant. This does not inherit dense dynamic split-K's undefined-empty-partial path. |
| SM103:883,991,1205,1222,1484 | K0 bypasses AB/SF work and MMA; accumulator generations skip it. | Accumulator/input clocks differ from output clock. Stage equations must be bound to their actual event stream; blanket per-output accumulator advance is invalid. |
| SM103 make_desc_and_call_mma:1883, layout helpers:1912–2092 and unrolled call sites | W96 instructions cross C256 AB chunks; eight current/next pairs, final-use positions2/5/7 exactly match the non-grouped SM103 source. | Reuse cross_buffer_operand_window@1 with unchanged mapping for active-K work. checks/circular_windows.py extracts both ASTs. Descriptor rebinding and K0 identity are separate premises, not changes to the window rule. |
| Both epilogues use shared C staging and bulk store groups | Output generations advance for all tiles; source-read waits protect reuse and final write observation remains distinct. | Reuse staged_async_output_store@1 after final registers are defined, including K0 zero registers. Changing which events advance the ring or the completion protocol needs a variant. |
| Both _compute_stages/_compute_grid methods | Stage counts derive from layout/shared-memory budgets; persistent physical grid is bounded by work and resident clusters. | Derived resource bindings, not independent free knobs. Existing output/input rules do not provide a measured residency or stage-cost model. |

## Replay and remaining boundary

[Grouped checks](checks/grouped-work-descriptors-result.json) cover1200 worker
domains and23112 queries across counts of1,2,31,32,33,64,65,97 groups and several
worker strides. The witness checks81920 distinct logical output coordinates,
two K0 output tiles, three input tuple updates and five C updates. Four worker-grid
choices exercise same-key update skips. Applying the K>0 input gate to C leaves
group0's descriptor while group1 owns the output; that wrong mapping is retained.
A separate model refuses descriptor mutation until every prior reader retires.

These are scalar source-equation and metadata checks. They leave warp instruction
lowering, exact lane/descriptor encoding, shape-cache register allocation, device
progress and input/output descriptor-reader retirement open. The positive
[bindings](inputs/grouped-binding-cases.json) state these premises explicitly;
schema checks reject missing monotonicity, immutable metadata, acquire or reader
retirement. They do not claim the source backend has discharged those premises.
