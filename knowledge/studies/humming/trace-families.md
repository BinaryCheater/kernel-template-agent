# NVIDIA mechanism traces

All paths are relative to the pinned Humming source. `explained` means the stated
source-level choice is recovered; it never means a CUDA implementation has been proved.
`partial` means a named rule fits some choices but its obligations or domain remain open.
`open` means the pinned rule cannot express the choice. Check commands and retained outputs
are indexed in `checks/README.md`; next evidence for unresolved rows is in `delta/gaps.yaml`.

## G01 — ordinary warp MMA

Witness: SM80 CUDA C++, N=128/K=128, FP16 A/C/scale, UINT4 B, explicit block=(16,128,64),
warp=(16,32,32), four stages, no TMA/warp specialization/Stream-K, group scale=32.
The JIT include is `kernel/humming.cuh`, selected by `kernel/humming.py:49–55`.

| ID | Source choice and concrete binding | Rule/field or gap | Outcome and evidence |
|---|---|---|---|
| D01 | `kernel/humming.py:362–384`: whole N/K blocks; N=128,K=128 satisfy block divisibility; padded M uses runtime bounds | `output_tiling@2`: tile_m=16,tile_n=128,executor=cta | explained; 8 warps = (16*128*64)/(16*32*32). M=17 has one full tile and a one-row tail; G06 guards the store. |
| D02 | `utils/context.cuh:89–96`: M/N/K warp counts=1/4/2; part K=16; two part-iterations per warp K32 | `reduction_partition@1`: group_size=32,num_groups=2,assignment=contiguous_parallel | partial; `composition.py` covers B for all 8192 points of one K64 stage; A and accumulator maps remain GAP01. |
| D03 | `kernel/humming.py:307–343`, `mma/wmma.cuh:82–99`: FP32 m16n8k16 instructions | `compute_instruction_selection@2`: mma_sync_m16n8k16_f16,rr,f32,2048 MACs | explained at source call boundary. 8 MMA calls per warp per K32; 64 calls cover 16*128*64 products. No emitted SASS claim. |
| D04 | `kernel/humming.cuh:129–207`: stage prologue then refill; register B alternates two buffers | `physical_storage@1`, `async_transfer@1`, `multi_buffer_rotation@2` | partial; four shared stages and private register double-buffering have different lifetimes. Do not assign the CTA-barrier rule's P to an asynchronous event schedule without a model (GAP02). |

SM75 uses a different FP16 MMA K8 and half-M cases (`kernel/humming.py:321–335`), with
synchronous copies (`utils/context.cuh:60`). SM80/86/87 use cp.async and target-specific
shared-memory menus; SM89 adds FP8 gates (`kernel/humming.py:411–485`). SM12x can use
native mixed FP8/FP6/FP4 operands (`kernel/humming.py:258–284`); that is a distinct type
binding, not evidence that software UINT4 decoding is used. These instruction variants
are partial under the current compute rule's explicit type menu (GAP03). FP16 accumulation
changes precision and register layout, requiring its own numeric check (GAP04).

## G02 — WGMMA and host selection

Witness: SM90, FP16 A/C and UINT4 B with per-channel scale; N8192/K1024, varying M.
`checks/policy.py` executes a closed slice of the actual host functions and dataclasses.
Its target inputs are SM90, NVIDIA, 132 SMs; they are analysis assumptions, not probe data.

| ID | Source choice and concrete binding | Rule/field or gap | Outcome and evidence |
|---|---|---|---|
| D05 | `tune/base.py:473–483`: power-of-two samples through 8, then multiples of 8 on SM90 | Family dispatch policy, not a Template legality constraint | explained; M9→16,17→24,33→40,65→72. `policy-result.json`. |
| D06 | `tune/sm90_policies.py:248–345`: specialized policies run first | GAP05 outside the recovered dense slice | explained for this slice: dense FP16 rejects compact A8, short-A8, FP8 block-scale, scaled-A8, overlap and packed-K specializations through their real early exits. No substituted result. |
| D07 | `tune/sm90.py:197–214`: WnA16, sampled M<=32, K<=1024, aligned N/K, no f16 accumulation/batch invariant → MMA | Instance policy selecting G01 | explained; M1/8 gets block M8; M9–32 gets block M16, block N128/K64, warp N32/K32, four stages. |
| D08 | `sm90.py:234–257`, `sm90_policies.py:909–1033`: remaining dense seed chooses WGMMA, warp-specialized TMA | `warp_specialization@1`, `async_transfer@1`, `compute_instruction_selection@2` | explained for M33/64/65/128: final block M40/64/72/128, N128/K64, warp M=block M,N32,K64; four stages, two CTAs/SM. |
| D09 | `sm90_policies.py:25–156`: fit register shape, residency and stages against 227 KiB | Resource constraint in source binding; GAP05 for unreplayed policies | explained as estimate: per-CTA bytes 38912/51200/55296/83968 for M33/64/65/128; multiplied by two fits 227 KiB. A compiled sizeof check is separate. |
| D10 | `tune/base.py:522–581`, `kernel/humming.py:709–756`: compress equal adjacent sampled configs into intervals; compile distinct configs | Family menu; native launch selection `launcher.cpp:85–95` | partial; the replay recovers ten chosen M points, not all interval endpoints or actual binaries (GAP05). |
| D11 | `kernel/humming_ws.cuh:91–152`: load and math roles, setmaxnreg; same Scheduler sequence | `warp_specialization@1`, `register_rebalancing@1` | partial; producer 128 threads, one consumer group for block N128/warp N32/K64; actual register budgets and spills require compiler evidence (GAP02). |
| D12 | `humming_ws.cuh:178–246`: issue/wait split, late scale prefetch, release after last WGMMA read | `per_slot_handshake@1`, `physical_storage@1` | partial; releasing before wait would destroy live operands. Source expresses this dependence; arrival/phase and tail races require GAP02. |

The generated WGMMA wrapper transposes the logical contraction: project B registers are
PTX A, project A shared descriptor is PTX B, and logical (M,N) is swapped in the PTX
shape (`config/mma.py:495–532`). Logical M40 therefore emits an m64n40k16 declaration.
Operand-source bindings must state which convention they use; full lane-coordinate
reconstruction is still GAP01/GAP03. `checks/specializations.py` recovers generated
SM75/80/90/100/120 type/instruction declarations directly from source, without compilation.

RS low-bit B uses register conversion; same-width SS B uses a shared descriptor. Packed-K
requires A8, even B bits, warp K128; group scales must be absent or at least warp K
(`kernel/humming.py:679–689`, `memory/s2r_loader/loader_b.cuh:30–61`). Synchronous WGMMA
selects `kernel/humming.cuh` instead of the role-split header. H 20 has separate candidate
ranking and overlap policy (`tune/sm90_h20.py`); indexed MoE disables A/C/input-scale TMA
and has row-index double buffers (`kernel/humming.py:631–663`). Multicast permits one axis,
and B multicast requires dense GEMM (`kernel/humming.py:666–677`). These variants are
accounted for as source-domain restrictions, with full host-policy replay open in GAP05
and asynchronous checks in GAP02. No performance superiority is inferred from the menu.

## G03 — UMMA, operands and TMEM

Witness: SM100, FP16 A/C, UINT4 B, block=(64,128,64), warp=(64,32,64), three stages,
CTA group=1, one dequant warpgroup, channel scales, no raw weight. Caller and builder are
the GEMM path; UMMA overrides the other headers (`kernel/humming.py:133–152,49–55`).

| ID | Source choice and concrete binding | Rule/field or gap | Outcome and evidence |
|---|---|---|---|
| D13 | `kernel/humming_umma.cuh:62–110`: load/readiness/issue/math/dequant roles; 256+128=384 threads | `warp_specialization@1` partially matches; GAP06 role topology | partial; source adds a dequant role and three different ready/free handshakes, which the two-role rule cannot bind completely. |
| D14 | `mma/umma.cuh:18–67`: four K16 iterations, operand columns=32, three operand buffers, one output group, M64 accumulator | `physical_storage@1` level=tmem; GAP06 allocation geometry | explained as derived sizing: 3*32+64=160, power-of-two TMEM allocation=256 columns; <=512. No TMEM footprint variable in the old warp role rule. |
| D15 | `humming_umma.cuh:136–150,208–232,342–372`: persistent stage and operand rings; issue retires operand buffers by tcgen05.commit | `per_slot_handshake@1`, `physical_storage@1`; GAP02 completion protocol | partial; three-slot example after five generations is stage2/phase1, not stage0/phase0; `mechanisms-result.json` detects a per-tile reset. This checks indices, not asynchronous safety. |
| D16 | `humming_umma.cuh:380–424`: wait accumulator, load output chunks, release accumulator and row-index lifetime independently | GAP06 accumulator/read-index ownership | partial; source distinguishes accumulator retirement from indexed epilogue completion; pairwise schedule requires a dedicated model. |

SS/raw weight removes dequant threads (256 total) and changes operand storage/scale
preparation; one/two dequant groups produce 384/512 threads in TS mode. CTA group 2 requires
block M divisible by16, N divisible by2*blockN, one CTA/SM and distributed barriers
(`kernel/humming.py:528–604`). Block N128/256/512 creates 1/2/4 output groups.
SM103/107 introduce K96 or doubled dense K paths in `mma/umma.cuh:19–25`; undocumented
E3M4/E0M3 is separately gated (`config/mma.py:43–79`), so major-version fallback does not
authorize every type. These source branches are partial under GAP03/GAP06; next evidence
is the exact descriptor, tensor-memory and barrier map for each alternative.

## G04 — MXMMA block scales

Witness: SM120, FP4 E2M1 A/B, FP32 accumulation, E8M0 scales at group 32,
block=(16,128,128), warp=(16,32,128), no zero point. The common GEMM header selects
`mma/mxmma.cuh`; `config/mma.py:120–146` derives MMA K64 and scale vector2.

| ID | Source choice and concrete binding | Rule/field or gap | Outcome and evidence |
|---|---|---|---|
| D17 | `mma/mxmma.cuh:19–48`: K part64, two iterations per warp K128; A/B scale registers | GAP03 compute scale operands | open; `compute_instruction_selection@2` has no block-scale instruction or scale-vector variable; do not bind it to ordinary FP16 MMA. |
| D18 | `mxmma.cuh:98–128`: scale-vector2 selects B byte0/2 with iter parity; A phase follows stage/warp K offsets | `physical_storage@1` plus GAP03 scale-fragment map | partial; source has distinct A and B byte ownership. End-to-end scale fragment indices still need exact PTX replay. |
| D19 | `kernel/humming.py:397–410,528–534,611–614`, `config/mma.py:149–172`: matching group sizes; no FP16 accumulation; zero point disables Stream-K; FP4 4X UE8M0 needs newer compiler | Source legality constraints, not empirical tuning facts | explained as guards; neighboring group 16 gives vector4 and invokes a different compiler gate. |

SM121 inherits SM120 format support but changes PDL and Stream-K/M-tile policy
(`tune/sm121.py:8–42`). Native mixed FP8/FP6/FP4 bypasses software decode; absent
input/group scales use encoded one. E4M3 vs E8M0 and scale vectors1/2/4 require distinct
bindings. These are GAP03; throughput and preferred tuning remain GAP05.

## G05 — dense, Stream-K and expert scheduling

| ID | Source choice and concrete binding | Rule/field or gap | Outcome and evidence |
|---|---|---|---|
| D20 | `scheduler.cuh:238–250`: dense raster, M blocks5,N blocks3,groupM4 | `block_rasterization@1`, `output_tiling@2` custom map | explained; final group has one M block. All 180 small raster domains are exact-once in `mechanisms.py`; using groupM4 at the tail would be wrong. |
| D21 | `scheduler.cuh:89–145,286–323`: separate data-parallel waves and Stream-K tail, alignment=lcm(group alignment,stages); reverse slice IDs | `linearized_work_partition@1` tile_major, `reduction_partition@1` | partial; source stripe alignment and reversed metadata are additional binding constraints, GAP07. Never claim plain ceil(total/grid) reproduces this scheduler. |
| D22 | `scheduler.cuh:154–234,326–423`: grouped counts [0,3,9], blockM4 → tile prefix [0,0,1,4]; contiguous offsets [0,0,3,12] | `runtime_work_descriptors@1`: device_kernel,pad_to_block; `indexed_gather@1` | partial; zero experts and per-expert ceil differ from ceil(total rows/blockM). `mechanisms.py` checks this witness; complete descriptor generation/GPU lifetime is GAP07. |
| D23 | `scheduler.cuh:276–280,386–423`: sorted route IDs [5,0,3,2], top_k2 → input rows [2,0,1,1] | `indexed_gather@1`; caller metadata contract | explained for index division; output remains indexed by route ID rather than input row (G06). Caller routing source is only partially archived, GAP08. |
| D24 | `scheduler.cuh:23–41,263–267`: A multicast/cooperative UMMA groups along N, B multicast along M | `cluster_multicast@1` only for actual shared operand; GAP06 for cooperative A halves | partial; cooperative UMMA loads distinct A halves, so it is not A multicast despite adjacent N tile grouping. |

## G06 — epilogue and combine

| ID | Source choice and concrete binding | Rule/field or gap | Outcome and evidence |
|---|---|---|---|
| D25 | `epilogue/pipeline.cuh:65–100`, `smem_reducer.cuh:13–116`: two K warps reduce partials before output | `smem_partial_combine@1`, `reduction_partition@1` | partial; group input scales can promote INT32 to float before combination; precision/cancellation requires GAP04. |
| D26 | `epilogue/smem_writer.cuh`, `gmem_writer.cuh:137–175`: fragment→swizzled scratch→coalesced int4 output | `staged_relayout@1`, `shared_resource_binding@1`, `vectorized_access@1` | partial; final row predicate for M17/blockM16 accepts local row0 only in second tile. N tails are padded in multiples of8; padded columns are not visible. Complete output layout replay is GAP01. |
| D27 | `epilogue/pipeline.cuh:108–133`: <=3 slices without TMA use equality lock; last resets | `ordered_global_combine@1` by_k_partition_index,f16,last_resets | partial; source slice IDs run in reverse K order. Acquire/release GPU visibility and residency are GAP02/GAP07. |
| D28 | `gmem_writer.cuh:167–172`, `pipeline.cuh:114–130`: native atomic and >3 slices, or TMA reduce-add | New `first_writer_atomic_combine@1` | partial; 873 completion orders over 2..7 partials and two generations preserve countdown in arithmetic replay; visibility and hardware completion remain open. |
| D29 | `pipeline.cuh:67–69,85–98`, `kernel/humming_ws.cuh:242–254`: wait TMA reads before scratch reuse, including final tile | `physical_storage@1.lifetime`, `staged_relayout@1.no_overwrite_before_read` | explained as source ordering; hardware completion and tail race checks are GAP02. |
| D30 | `arith/epilogue_arith.cuh`, `pipeline.cuh:139–171`: channel/tensor scales and bias at output; group scales remain inside their K group | `operator_fusion@1`: scale_and_cast or linear_combination | partial; unequal-group witness 160 vs invalid postscale128 rejects moving arbitrary group scales through the entire K sum. Repeated bias/scaling of split partials needs GAP04. |

## G07/G08/G09 — weight representation composition

The detailed packer and bitstream trace is in `trace.md`. `composition.py` now composes
G08's A16/B4 matrix layout through the order-preserving legacy G2S copy, the half-group
UINT2 S2R load (`memory/s2r_loader/loader_b.cuh:115–145`), nibble-pair extraction
(`datatype/dequant.cuh:10–38`) and the FP16 m16n8k16 B fragment. Each of 8192 B coordinates
is correct and unique for a block N128/K64. The source witness (17,43) reaches N-warp 0,
K-warp 1, iteration0, lane 5, first N8 instruction within fragment 1, high K8 half. Half mantissa decode
recovers q-z exactly for all 16 codes at z5/8/13. A source-derived unequal scale witness
distinguishes groupwise from whole-reduction scaling.

| ID | Source choice and concrete binding | Rule/field or gap | Outcome and evidence |
|---|---|---|---|
| D31 | `datatype/dequant_single.cuh:25–46`: FP16 magic base, UINT4 → half, zero point8 | `fused_dequantization@1`: magic_number_lop3,symmetric_folded,group_size32,per_group_before_mma | explained for conversion and B layout; floating error after scale multiplication is GAP04. |
| D32 | `datatype/dequant.cuh:60–126`, `kernel/process.cuh:57–84`: odd widths 3/5/6/7 compressed via different masks; different warp-N split paths | GAP09 prepack/decode domain | partial; existing prepack pack_bits menu4/8/16 and INT4 decoder obligation cannot express all these fields. No untested version widening proposed. |
| D33 | `kernel/process.cuh:242–265`: optional INT→FP and ZP preprocessing, mini-block transpose | GAP09 numeric/layout composition | partial; preprocessing is not just a permutation. Its inverse depends on dtype and zero-point convention. |
| D34 | `datatype/dequant_single.cuh:85–184`, `arith/exp_offset.cuh`, `datatype/dequant_native.cuh`: mini-float/e0mX/native conversions and exponent compensation | `operand_precision_conversion@1` concept; GAP09 exact format fields | partial; e0mX has sign-magnitude integer magnitude, unlike an IEEE exponent-bias conversion. Full range/rounding and native conversion pairing remain open. |
| D35 | `kernel/humming.py:238–256`: native ldmatrix dequant gated on SM>=90/CUDA>=13.4; otherwise patch CUBIN placeholders | Realisation selection with GAP03/GAP09 | explained as source route, not established machine instruction execution. Undocumented encodings have a separate target allowlist. |
| D36 | `kernel/process.cuh:277–319`: expert and padded-K offsets, packed-K output branch | `offline_operand_prepack@1` partial source binding; GAP09 | partial; recovered N64/K64/A16/B4 identity does not generalize to packed-K, native layout or expert tails. |
| D37 | `kernel/pack_weight.cuh:4–103`: continuous codes vs fragment order | New `bitstream_packing@1` | partial; all code-bit and CTA-boundary witnesses pass; CUDA inactive local initialization is GAP10. |

## G10 — quantize and standalone decode

| ID | Source choice and concrete binding | Rule/field or gap | Outcome and evidence |
|---|---|---|---|
| D38 | `ops/weight.py:67–129`, `kernel/quant_weight.py:64–75`: complete128-value FP16 group→UINT4, integer ZP, one warp | New `groupwise_affine_quantization@1` group_size128,code_bits4,zero_point=integer | partial; codes repeat 0..15 with scale 1/z8 in `mechanisms.py`; scale metadata follows group index. Boundary parameters must match the requested numerical policy. |
| D39 | `kernel/quant_weight.cuh:95–118,225–258`: symmetric negative scale, one-bit special divisors, e0mX, floating-ZP and UE8M0 ceiling | GAP11 numeric quantizer variants | partial; six integer halfway cases and five positive E8M0 scale boundaries are checked. These do not settle all policies, zero/nonfinite groups or overflow. |
| D40 | `kernel/dequant_weight.cuh:13–50`: 32 codes/thread, guarded visible load/store; e0m3 code13→-5 | `operand_precision_conversion@1` conceptual reuse; GAP09 | explained e0mX branch by sign/magnitude formula; mini-float bit rebias and inactive local values need range and compiled checks. |

## G11 — input operation, layout and phases

Witnesses: (a) SM80, one FP16 token K128 → dynamic-token INT8; V4,32 threads,
one token/CTA; (b) normalized Hadamard H 128/V4 then H 256/V4; (c) FP4 group-token
input with tiled group scale generation and separate finalizer.

| ID | Source choice and concrete binding | Rule/field or gap | Outcome and evidence |
|---|---|---|---|
| D41 | `kernel/process_input.py:147–194`: fused vs CollectAbsmax/Quantize, or group-token finalizer | `split_combine_kernel@1`, `reduction_partition@1` partial; GAP12 scale intermediate contract | explained branch generation: dynamic-token+two_stage builds two instances of the same device entry with different phases; tiled group-token adds a different finalizer entry. |
| D42 | `kernel/process_input.cuh:130–167,191–223`: activate→Hadamard→quantize, keep intermediates in registers; raw-copy bypass | `operator_fusion@1` for existing scale/cast scope; activation menu GAP12 | partial; Split/Interleaved gate/up addressing differs (`activation.cuh:104–150`). Choice of activation is a numerical contract field, not a free scheduling parameter. |
| D43 | `kernel/process_input/hadamard.cuh:8–100`, `context.cuh:81–83`: H power-of-two<=512, complete blocks; local then shuffle/shared stages | New `butterfly_linear_transform@1` | partial; all basis vectors at H 4/32/128/256/512,V4 match Walsh signs; cross-warp scratch synchronization and rsqrt rounding are open. |
| D44 | `kernel/process_input/quantization.cuh:144–295,298–314`: maxima reduce locally then warp/shared, target max127 for INT8 | `cross_lane_reduction@1`, `reduction_partition@1`; quantizer format GAP11 | partial; positive absmax permits bitwise max on SM80+, older targets use shuffles. Group bounds and scaling must survive invalid rows. |
| D45 | `process_input.cuh:170–209,238–301`, `quantization.cuh:23–115`: M3BF16 group scale, power-of-two token scale, E4M3 ratio; group stride padded to four bytes | GAP12 scale factorization and layout | partial; finalizer reads rounded group scales, emits token scale and ratios, and clears padding. Exactness holds only where the ratio is representable; underflow stays an obligation. |
| D46 | `kernel/process_input/context.cuh:109–155`, `thread_task.cuh:20–39`: normal/grouped-mask/scatter, valid tokens, int32/int64 descriptors, zero vs skip | `runtime_work_descriptors@1`, `indexed_gather@1`, `boundary_predication@1` | partial; invalid route must not suppress a valid route's shared input read; masked rows still participate in collectives. Caller descriptor exactness is GAP08/GAP12. |
| D47 | `process_input.cuh:104,228–231,257,297–300`, `forward.py:37–49`: PDL wait and launch-dependent signals | GAP02 cross-kernel ordering | explained as source placements. SM121 always prefers PDL, SM120 conditions it on small-M/wide-N; no measured benefit or dependency correctness claim. |

## G12 — fused E8M0 / MXFP4 rebase

| ID | Source choice and concrete binding | Rule/field or gap | Outcome and evidence |
|---|---|---|---|
| D48 | `transform.py:242–289`: clamp group scale range, change scale origin, compensate secondary scale, invoke code rebase | GAP13 coupled numerical representation | partial; this can re-quantize weights, so it is not merely a lossless scale factorization. Need the original and effective code*scale contract together. |
| D49 | `kernel/process_mxfp4.cuh:6–37,40–69`: 32 FP4 codes/group; delta1; canonicalize code8; nearest ties choose larger magnitude | `operand_precision_conversion@1` partial; GAP13 | explained code map for all 16 codes and deltas0..3 against an independent E2M1 table. Code1/delta1 distinguishes tie-away from tie-to-zero. `kernel/process_mxfp4.py:18` disables fast math, preserving the source's subnormal arithmetic route. |
| D50 | `process_mxfp4.cuh:41–42`, builder shape assertions | `boundary_predication@1`, `vectorized_access@1` | explained source tail: index>=num_groups returns, active thread reads/writes one uint4. Full transformed-weight error needs GAP13. |

## G13 — isolated compute benchmark

| ID | Source choice and concrete binding | Rule/field or gap | Outcome and evidence |
|---|---|---|---|
| D51 | `ops/bench.py:8–41`, `kernel/tops_bench.py:38–76`: SM80 FP16 MMA m16n8k16, repeat 65536,unroll 64 | `compute_instruction_selection@2`, `vectorized_access@1` not required for a compute-only witness | explained arithmetic: 4096 FLOPs/instruction/warp,32 warps/CTA,2*SM_count CTAs. This is the numerator of the benchmark, not measured throughput. |
| D52 | `kernel/tops_bench.cuh:24–46`: WGMMA fence/commit/wait; MXMMA scale operands; ordinary MMA fallback | Existing instruction Realisation or GAP03 | partial; default SM90 uses WGMMA, other targets MMA; explicit MXMMA differs. Benchmark registers are not initialized as a numerical test and the conditional output is not a correctness oracle. |
| D53 | `ops/bench.py:43–45`: triton.testing.do_bench timing→TOPS | GAP14 empirical execution | open; no timing has run and no performance Finding is proposed. UMMA is not established as a supported benchmark route merely because its enum exists. |

## Transfer and lineage

The README explicitly credits Marlin, DeepGEMM, lmdeploy and CUTLASS. Low-bit prepack,
register decode and Stream-K share mechanism identities with the Marlin seed; source
lineage alone does not establish equivalent domains. TMA/warp roles and per-slot barriers
match the FA3-derived candidates only at their stated contracts. UMMA's TMEM and role
topology and MXMMA's block-scale operands are recorded as gaps rather than forced into
those older fields. No pending addition from another study is a dependency: the pinned
reviewed index was empty. No old version is overwritten.
