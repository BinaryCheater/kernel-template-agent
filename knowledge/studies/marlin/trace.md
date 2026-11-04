# Marlin caller, prepack and decode

Pinned source paths are relative to [the retained tree](inputs/sources.json). The trace follows Python Layer/mul, C++ binding, all ten compiled
specializations, packing, shared/register layouts and striped output completion. No CUDA build or execution was performed.

## Contract and concrete instance

The Python Layer requires K divisible128, N divisible256 and groupsize=-1 or128.
The -1 setting becomes one scale row over K. It allocates packed int32 B `[K/16,2N]`,
FP16 scales `[K/groupsize,N]`, and zero workspace `N/128*16` int32 entries. Forward
flattens all activation leading dimensions into M and views the result back. Packing
expects a fake-quantized FP16 Linear weight and corresponding scales; replacement
helpers create empty layers and do not themselves quantize pretrained weights.

The Python mul doc's scale shape mentions M/groupsize, but actual code and scale
ownership use K/groupsize. Its max_par default16 also differs from C++'s default8;
Python always passes its value. C++ derives groupsize from s.size(0), validates divisibility
and workspace element count, then passes raw pointers and A's device/current stream.
It does not validate all tensor dtypes, devices, strides, B/C shapes or workspace zeros.
Those are caller premises, not guaranteed by the thin binding. Zero dimensions may
return without writes; K0 is not specified here as a zero-filled GEMM result.

Take M16,K256,N256,groupsize128,max_par16 and auto thread tiles. The launcher chooses
thread_k128/thread_n128,256 threads,four input stages and fixed96KiB dynamic shared
storage. The compiled witness is Marlin<256,1,8,8,4,8>. There are two K tiles and two
N tiles before striped partitioning; grid defaults to device SM count. The source
compiled menu has10 variants: M-block count1 with N/K blocks8/8 or16/4, M-blocks2/3/4
with16/4, each with per-channel or group128 scaling. A user tile setting outside that
menu can pass shape division yet return "no kernel implementation". Build uses
CUDAExtension over the C++ binding and one CUDA source; there is no explicit host
architecture gate. Selected instructions are cp.async, ldmatrix and m16n8k16 FP16 MMA.
Instruction availability/build target behavior remains an unexecuted boundary.

## Choices and reuse

| Source choice | Pinned existing rule and binding / gap | Evidence |
|---|---|---|
| `marlin/__init__.py:40–68,113–139`: tile transpose,1024-entry permutation and eight-nibble interleave | `offline_operand_prepack@1`: tile_rows16,tile_cols64,element_order=mma_fragment_interleaved,pack_bits4 | All1024 permutation entries are unique;229376 uniquely tagged coordinates invert the host layout. B-fragment composition is checked below; A/output replay closes the logical mapping. |
| Same pack function: grouped vs per-channel scales use64- vs32-entry permutations | Same rule's scale_alignment obligation | Both permutations are bijective but differ. They cannot be interchanged based on equal byte values. Both grouped and per-channel scale-read compositions are checked below. |
| `marlin_cuda_kernel.cu:116–160`: two lop3, FP16 subtraction and fused multiply-add | `fused_dequantization@1`: method=magic_number_lop3,zero_point=symmetric_folded | Exhaustive65536 selected-nibble combinations decode q-8 exactly. Calling dequant(q) and dequant(q>>8) restores all eight pre-interleave slots. This checks bit/half arithmetic, not emitted instructions. |
| Device `:442–464`: group scales multiply decoded FP16 B before MMA | Same rule: scale_site=per_group_before_mma,group_size128 | FP16 scale multiplication is a numerical boundary; decoded integers are exact, arbitrary scaled products need tolerance checks. |
| Device `:659–678,588–592`: only last contributor fetches per-channel SF; cast accumulator before multiply | Same rule: scale_site=per_channel_in_epilogue,group_size=-1; numeric contract gap | The source is round16(sum)*scale in half arithmetic, not one final round of an FP32 scaled sum. The retained witness distinguishes them. No unrestricted scale-motion equivalence is asserted. |
| Host `:754–816`: M chunks of at most64 rows, parallel full64-row problems and a final tail | output_tiling@1 and dispatch policy, block_m16/32/48/64; parallel full64 chunks and a separate tail | Auto tiles change at M16; incomplete user tile pair resets both auto values. Grouped stripe length rounds to whole scale groups. Workspace must match max_par and remain nonoverlapping across invocations. |

## Packing equations and checked boundary

For logical `(k,n)`, host tiling first produces flattened position
`floor(k/16)*(16*N) + floor(n/16)*256 + (k%16)*16 + n%16`.
Each1024-entry region then uses the source permutation, followed by packing eight
codes per32-bit word in increasing nibble order. The two decoder calls extract
packed nibble positions0/4/1/5 and2/6/3/7. Composing that with host order
0/2/4/6/1/3/5/7 recovers original fragment slots0..7. Original coordinate tags are
checked before four-bit packing, so repeated nibble values cannot hide a permutation.

The scalar decoder interprets the source half constants directly with standard-library
binary16 packing. Integer extraction is exact on this complete selected-nibble domain.
A separate witness uses accumulator1.00048828125 and scale3: casting to half before
scaling differs from scaling in FP32 then casting. This source policy must remain in
any admitted per-channel numerical realization.

[Checks and retained result](checks/README.md) establish input retention, host inverse,
decode arithmetic and that rounding boundary. The starting Marlin reference contains
more proposed rules/chains; it is guidance, not proof of this pinned revision. Its
symbolic chain bindings and old instruction-selection identity must be reconciled to
current schemas before reuse. No new Template or rule version is introduced in this
slice. Resume with actual packed-word/scale ownership, stripe overlaps and ordered
partial result commit, then tests/GPTQ caller domains and whole-library closure.

## Stripes, fragment coordinates and ordered commit

For K1024,N512,M16,thread_k128,thread_n128 and grid5, the linear domain has32 units
(K tiles inside four N columns). CTA stripe width is ceil(32/5)=7. In the first column,
CTA0 owns K tiles[0,7) and CTA1[7,8), but their slice_idx values are1 and0 respectively.
The source commits higher K ranges first. The same CTA may continue into another N
column or another parallel64-row problem; each column's contributor metadata must be
computed from the intersection with the stripe rather than assuming all groups have
uniform lengths.

`init_slice` at device lines245–279 calculates count and reverse index from stripe
starts. The replay independently enumerates interval intersections for1800 domains,
including one/two-stage quantization groups, partial final stripes, parallel problems
and mostly idle grids. All29341 slice segments agree on count/order and exactly-once
unit ownership. Grouped stripe width and K tiles are multiples of group_tiles, so a
stripe never begins inside a quantization group for the compiled group128 menu.
`linearized_work_partition@1` with units_per_cta7,order=k_major_within_column,grid5
expresses the witness. Applying the generic reduction_partition's uniform group size
alone omits clipped first/last stripe offsets; retain the stripe decoder as context.

The actual B global/shared read map is now composed with host packing. Each256-thread
CTA reads int4 vectors; two register substeps cover its K64 or K128 tile. Packed
vector indices recover logical16×64 tiles, then the decoder's nibble offsets produce
the two B fragments. The [NVIDIA PTX fragment reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float)
places FP16 B elements at rows determined by lane residue and element pair, with
column determined by the lane's four-thread group. Composing those coordinates with
Marlin's word/fragment offsets matches262144 unique-coordinate checks across both
compiled K/N geometries and multiple N tiles. Grouped scale reads match4096 column
bindings using the separate64-entry scale permutation. The reference used is recorded
in [ptx-fragments.json](inputs/ptx-fragments.json). The final A/output replay below completes logical coordinate coverage; no check executes MMA.

A shared-memory transform is the whole int4 index XOR its row index, according to C++
operator precedence. The replay checks1536 locations across all compiled M/K shapes:
it is a permutation, but is not generally an involution once row bits reach higher
address bits. Correct composition is store[f(logical)]=value, then read[f(logical)],
not applying f twice to an already transformed address. Register fragments double
buffer by k%2. Shared stages rotate4 with three prefetched groups and wait_group2 plus
CTA synchronization before reads. Empty tail fetches still commit an async group,
maintaining wait accounting. Group scales for K64 live across two adjacent stage slots;
the160-slot replay agrees with the source's grouped SF ring-base formula.

After draining async input transfers, source `thread_block_reduce` uses a shared-memory
log tree among K-splitting warp groups. The shared storage aliases the input pipeline
only after the drain; accumulators remain FP32. Global combine then waits for lock
value=slice_idx. First contributor writes an FP16 partial without reading old C;
intermediate contributors add in FP32 and round partial C to FP16. The last contributor
adds prior C but skips the intermediate store, resets the lock and alone performs the
final fragment permutation/output store. Per-channel scales are fetched only by that
last contributor; grouped scales already participated before MMA.

`ordered_global_combine@1` binds order=by_k_partition_index,partial_dtype=f16,
lock_reset=last_resets, with "partition index" explicitly the reverse-K source index.
A scalar witness with increasing-K partials[-2048,1,2048] yields0 in the source order,
but1 in increasing-K order due to FP16 partial storage. Its associative exact-real
premise is not a claim that these floating-point policies are interchangeable.
Numerical admission must retain the actual order and storage rounds.

The lock resets before final output stores. It therefore does not protect concurrent
reuse of C/workspace by another invocation; external invocation completion and stream
ordering remain premises. Nonfinal release uses an acquire/release fence then relaxed
atomic increment; final reset is an ordinary zero store. Actual visibility, shared
reduction races, CTA residency/progress and reset lifetime need device evidence. No
GPU synchronization proof is inferred from scalar lock states0/1/2/0.

The [partial concrete chain](delta/chain.yaml) reuses prepack, stripes, decode and
ordered combine; shared fragment pipeline and intra-CTA reduction are explicit external
edges. Its four identities/binding domains match the pinned pool. [All retained checks](checks/README.md)
cover source equations and declared boundaries. No new Template is proposed. The remaining logical maps and caller reconciliation follow.


## A fragments, output and caller closure

| Source choice | Reuse or disposition | Evidence |
|---|---|---|
| `marlin/marlin_cuda_kernel.cu:287–313,353–358,436`: XOR-swizzled int4 addresses feed ldmatrix.x4 | Existing prepack/fragment interface plus explicit A shared-layout context; no new rule | 45056 uniquely tagged A-fragment coordinates across all five compiled geometries. |
| Same device source `:465–509`: CTA-local tree before global completion | `smem_partial_combine@1`: topology=log_tree,combine_op=add,result_to=one_warp,participants=4 for N128 or2 for N256 | Participants are K-splitting warp groups, each containing the N-distributed warps. Source branches and barriers traced; floating-point tree order and GPU race checks remain obligations. |
| Same source `:573–617`: fragment half2 stores to padded shared rows, then coalesced int4 output | Output-layout glue; source stride=2*Nblocks+1 int4 entries, only live M rows written | 1020928 live output coordinates, unique and complete for every valid row count of each compiled shape; no write to padded rows in the replay. |
| Same source `:588–603,659–678`: last-contributor scale load/select | Existing fused-dequantization epilogue binding and scale-alignment obligation | 21504 column bindings compose the32-entry host scale permutation with int4 register loads and half2 selection. Preserve cast-before-scale arithmetic. |
| Host `:784–819`: four M blocks per chunk, parallel full64 chunks | Output tiling plus host dispatch policy | 4100 domains: M0..1024 with max_par1/2/8/16 cover each live row once. max_par must be positive: M65/max_par0 leaves loop index unchanged. The binding does not reject this input. |
| `test.py`: tile/tail/group/error tests | Caller evidence, not execution evidence | Both tile geometries, M tails and group128 appear. `test_llama_shapes` returns immediately. Broad exception tests can fail an earlier guard. Quantization helper overwrites a valid Layer's buffers/shape to exercise direct-mul shapes outside Layer constructor constraints. |
| `bench.py`: timed launches with arbitrary packed B and zero scales | Timing scaffold only; no performance or correctness Finding extracted | Benchmark and test helpers allocate default-FP32 zero workspace; zero bit patterns satisfy the lock's initial contents but do not establish a typed interface. Production Layer allocates int32. |
| `gptq/llama2.py:101–103,209–223`, `quant.py:18–76`, `gptq.py:55–145` | Calibration/quantization caller; no additional NVIDIA device body | configure defaults to symmetric quantization. Four-bit symmetric zero point is8; pack receives scales alone. Generic asymmetric quantizer settings cannot be passed through this scale-only path as equivalent semantics. |
| `gptq/eval.py:10–30`, `datautils.py`, remaining model/calibration helpers | Framework/data integration | Checkpoint suffix selects per-channel versus128-group Layer; model replacement/load and dataset/calibration code delegate computation, adding no custom CUDA family. |

The [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix)
provides the logical ldmatrix row-address and register-fragment relationship. The replay
composes that relationship with the source XOR mapping and the m16n8k16 A fragment
coordinates. The four matrices represent row halves and K halves of one16×16 A
fragment. Output checks independently identify each source accumulator coordinate and
follow shared writes into global int4 stores. These are source-equation checks;
compiled lowering and hardware behavior remain outside their evidence.

All18 retained files now have an inventory disposition. The one extension entry and
one CUDA family generate ten compiled specializations (five geometries × two scale
modes), all covered by the common trace and the stated differences. The extraction
is complete. No new Template/ProofOutline pair or Realisation is proposed. Existing
reference rules remain conditional on numerical contracts, caller-valid storage,
workspace lifetime and device synchronization/progress; independent admission remains
pending. The four-node Chain is deliberately partial: shared pipeline, intra-CTA tree
and final-output layout are source-traced external edges, not a whole-kernel proof.
