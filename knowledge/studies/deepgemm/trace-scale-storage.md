# Scale storage and SM100 grouped execution

Read with [the first trace](trace.md); sources and direct submodule revisions are retained in [the source manifests](inputs/submodule-sources.json). All paths below are revision-relative unless prefixed CUTLASS or DeepJIT. This pass follows source-selected CUDA paths for SM90a and SM100f. Compile target defaults are read from DeepJIT, not observed on a device.

## Exact scale values and physical grouping

`csrc/apis/layout.hpp:12–58` first validates the input SF shape/strides. On SM90 it transposes `(gran_mn, gran_k)=(1,128)` FP32 scales to MN-major storage; `(128,128)` scales are validated without packing. On SM100 it broadcasts FP32 scale groups along MN and packs exponent bytes, or validates already-packed int32 storage for gran_mn1/gran_k32 or128. Independent recipes allow gran_k_a and gran_k_b to differ in non-K-grouped GEMM.

The source bit extraction preserves the value of **positive normal FP32 powers of two**. It is not a rounding operation. Every such scale has bit pattern `e<<23`, e1..254, and UE8M0 decodes e to `2**(e-127)`. All254 values are checked against a decoder independent of the source packing shifts. The source sign/mantissa mask also accepts FP32 zero and positive infinity; their packed codes0/255 decode to a positive minimum/NaN. This is why [power_of_two_scale_encoding@1](delta/power_of_two_scale_encoding@1/template.yaml) requires a stronger live-value premise than the device mask. Padding code0 is a finite storage fill and must not be mistaken for a zero-valued live scale. The retained CUTLASS decoder establishes these endpoint meanings (`include/cutlass/float8.h:1138–1214`).

Four encodings at K-scale indices4q..4q+3 occupy byte offsets0,8,16,24 of an int32. For non-K-grouped MN=6, SF_K=5, the output shape is `[6,2]`, strides `[1,8]`; two leading groups have group stride16. The second word of each row has one live byte and three padding bytes. The scalar witness tags every row, K byte and group independently. Broadcast maps logical row r to input scale row floor(r/gran_mn); its numerical ownership is preserved separately from physical transpose.

K-grouped packing uses a different row arena: each group starts at a new packed word row, including a partially occupied last row. For input SF group lengths `[1,3,4]`, packed row starts are `[0,1,2]`, not the starts obtained by packing all8 scales together. Each group's input SF rows cover its aligned K allocation. The source `num_padding_sf_rows` tracks the difference between input-row and packed-row prefixes. The check compares that cumulative-padding implementation with independent per-group packing for216 domains. The existing Humming `bitstream_packing@1` requires complete32-code groups; its declared domain does not express these ragged reset points. Keep a packing-domain extension open; no historical identity is changed here.

The SM100 UTCCP pre-transpose rewrites128 int32 words from `i*32+lane` to `lane*4+i` (`impls/sm100_fp8_fp4_gemm_1d1d.cuh:462–489`). All128 uniquely tagged words and their bytes retain ownership under the inverse mapping. The hardware UTCCP-to-tensor-memory data-path mapping is a subsequent boundary, not established by a correct shared-memory permutation alone.

## One SM100 witness and neighboring variants

Witness: SM100 CUDA, pure FP8, K-major A/B, BF16 identity output, no C, K1152, selected source-menu tile M128/N256/K128, cluster1, gran_k_a32 and gran_k_b128. Configuration is supplied for source substitution; no heuristic winner or timing is claimed. Host API enters arch_major10 with packed int32 SF, passes independent grains into `SM100FP8FP4Gemm1D1DRuntime::compile_and_launch`, and instantiates `sm100_fp8_fp4_gemm_1d1d_impl` (`csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:35–89`).

A consumes a new packed scale row every128 K (four32-element scales). B consumes one packed row every512 K and uses the stage-in-group byte0..3 for its four128-element scales. At K offset160, A selects logical scale5 and B scale1. Applying A's selector to B would silently select a wrong scale. Nine K blocks include two complete B-scale reload intervals and a new third interval. The scalar check compares all selected logical rows to floor(global_k/gran_k), including unequal A/B grains.

Pure FP4/FP4 selects packed shared-memory elements and MXF4: BLOCK_K256, UMMA_K64, grain32 on both operands, and two SF subblocks. Each K64 MMA consumes two32-element scales; offsets0,64,128,192 select pairs `[0,1]`, `[2,3]`, `[4,5]`, `[6,7]`. Mixed FP8/FP4 uses unpacked FP4 shared-memory storage with FP8's K32 instruction family. The byte packing factor must match between shared operands. The source's MMA choice is not a new numeric Template for post-partial scaling: this hardware instruction consumes scales internally.

For instruction semantics, consult [NVIDIA PTX block scaling](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma-block-scaling). The source PTX wrapper chooses one/two CTA forms. MXF4 uses `.block32` on CUDA12.9+ and its `.scale_vec::2X` spelling on earlier toolchains; these express two scale factors over K64. This is a source-selected wrapper path, not emitted PTX or SASS.

K-grouped variants additionally require equal A/B K grains and k_alignment divisible by BLOCK_K. Their `Scheduler` receives `kSFKSpan=gran_k_a*4`, so packed-SF prefixes follow independent group packing. M-grouped masked uses capacity-based tensor maps as on SM90. M-prefix swap-AB with ensure_zero_padding=false derives an effective M aligned to16 for the last tile, updates UMMA N and load extent, and still requires the allocation/consumer contract to distinguish valid rows from aligned work. Batched variants keep data batch index in the tensor map but concatenate SF groups along packed K; contiguous M variants select B's expert while retaining A's global row coordinate.

K-empty groups deliberately execute at least one K block (`max(1, ceil_div(current_shape_k, BLOCK_K))`); the epilogue selects zero product values after reading the initialized accumulator. With accumulation, TMA reduction adds that zero into D. This source protocol maintains matching barrier progress; initialization, allocation and zero-row/group cases need a CUDA check before runtime correctness is claimed.

## Shared tensor-memory columns and early release

The source fixes two accumulator stages. N256 with SF_A/SF_B each4 columns would require520 columns; capacity is512. It derives overlap8 and stage starts0/248, with SF columns504..511. The two logical accumulator stages share columns248..255. Capacity alone would not justify this alias: the preceding epilogue must read those columns before the next stage's first MMA writes them.

Stage0 reads high columns first; stage1 reads low columns first. Both therefore finish their overlap reads first. A separate overlap barrier releases those eight columns once their loads have reached registers, while the whole-stage empty barrier waits for all256 columns. The next iteration's scale UTCCP can occur before the overlap wait, because the scale columns do not alias either accumulator. Only the first UMMA waits on the preceding stage's overlap barrier; same-stage reuse waits on the whole-stage empty barrier.

Reuse `physical_storage@1` on the overlapping subregions with last_read before next_write; use separate `per_slot_handshake@1` instances only where its full/empty signaling domain matches. The source's early subset signal is additional family glue, not permission to release the entire stage. No new storage Template is needed merely for a different last-read point. `multi_buffer_rotation` is seed **version2**; its CTA-wide event model is not silently treated as a proof of this asynchronous instruction protocol. Source stage counts above8 are also outside its variable domain. These are explicit binding gaps.

## Decision ledger

| Source and concrete binding | Rule field / gap | Evidence or remaining obligation |
|---|---|---|
| `apis/layout.hpp:35–56`: arch/type/granularity SF transform menu | Workload layout gate | FP32 normal-scale encoding vs int32 validate-only are separate paths. |
| `impls/smxx_layout.cuh:174–194,267–294`: exponent extraction | `power_of_two_scale_encoding@1.site=layout_transform` | All254 live exponents; zero and infinity endpoint counterexamples. |
| Same file`:95–105,175–194`: four-byte packing plus MN-major output | Storage grouping gap; `physical_storage@1.level=gmem` | Unique row/group/K tags and tail bytes; no arbitrary-length reuse of Humming bitstream@1. |
| Same file`:224–270`: K-group prefix and per-group padding | Group storage contract gap |216 grouped cases; flat packing rejected. |
| `sm100_fp8_fp4_gemm_1d1d.cuh:88–101,259–273,416–428`: independent SF reloads/selectors | Metadata ownership and `compute_instruction_selection@1` partial match | Four grain pairs replayed; this rule's instruction menu lacks scaled-UMMA fields, so not a complete binding. |
| Same file`:51–85,409–440`: FP8/mixed vs FP4/FP4 | Backend Realisation boundary | K32 vs K64/two-scale selectors; packed FP4 requires K-major/grain32. |
| Same file`:114–127`: 2-stage alias and capacity formula | `physical_storage@1`, source-derived alias extent |15 interval domains; alias has no effect on the disjoint SF columns. |
| Same file`:323–405`, `epilogue/sm100_store_cd.cuh:98–132`: subset/full release | Lifetime composition; additional early-release barrier gap | Six alternating generations per domain; releasing before final overlap read leaves a specific unread column. Hardware visibility remains open. |
| Same file`:166–190,275–295,480–493`: transaction and thread arrivals | `per_slot_handshake@1` partial match | SF full and A/B full use different barriers; transposer arrives only after its writes/fence. Two-CTA proxy fence is an upstream TODO, retained as an obligation. |
| `csrc/runtime/jit.hpp`, DeepJIT `backend/cuda/{device, options, backend, kernel}.hpp` | Compile and launch infrastructure | Device-derived architecture string defaults SM90a/SM100f; nvcc receives `--gpu-architecture`; launch uses cuLaunchKernelEx with supplied grid/block/cluster and dynamic shared memory. No actual compilation run. |
| DeepGEMM `mma/sm90.cuh:14–32` → CUTLASS `mma_sm90_gmma.hpp:13839–13898` | SM90 candidate Realisation | Selected N96 inline WGMMA has48 FP32 lane accumulator registers; scaleD zero/one implements reset/continuation. Source-only boundary. |

Evidence: [scale storage](checks/scale-storage-result.json), [SM100 schedule](checks/sm100-schedule-result.json). Tests are finite scalar substitutions and interval arithmetic. Outstanding source-to-hardware mappings and numerical/GPU checks are named above. Remaining library families stay in the inventory queue.
