# GeForce dense GEMM — entry and resource/output slice

When a fixed-shape GEMM has many independent tiles, consider static persistent
work assignment and a shared output staging ring. Retrieve
`persistent_tile_scheduler@1.fetch=static_stride` for work ownership and
[staged_async_output_store@1](delta/staged_async_output_store@1/template.yaml)
for output source retirement. Preserve exact input type and stage-capacity gates;
an earlier width check is not evidence that the selected arithmetic instruction
accepts that type.

Scope: the single `dense_gemm.py` under
`examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/` in the
[retained tree](inputs/sources.json). [Inventory](inputs/geforce-dense-variants.json)
pins its functions and methods. The source entry is now extracted, including the scheduler/layout correction below.
Native layout/precision/resource obligations remain explicit; source closure does
not admit a working native capability.

| Source decision | Concrete binding | Old field or implementation boundary |
|---|---|---|
| Parser:115–208 | Seven CTA tiles:64×64×64,64×128×64,128×64×64,128×128×64,128×256×64,128×128×128,256×128×32. Four problem dimensions; positive epilogue-stage count without an upper bound. A major K/M, B major K/N, C major N/M. | Compiled CLI menu, not hardware completeness or measured optimum. Generic dtype parser adds no per-operand arithmetic support gate. Direct `run` bypasses parser-only checks. |
| Constructor:212–256, setup:258–285 | Cluster1, occupancy1, eight MMA warps from atom layout(4,2,1), one active DMA warp and three idle warps in a384-thread CTA. MMA atom16×8×16; tiled permutation64×32×16. | Cooperative same-item computation, not striped pingpong. Existing `warp_specialization@1.same_sequence` fits this role relation conditionally; no `interleaved_consumer_streams@1` instance is needed. Register budgets232/40 are source choices, not a proved occupancy result. |
| `__call__`:361–381 then `_setup_attributes`:258–275 | Width checks accept equal-width8-bit types, including unequal FP8 formats. Actual op is always `MmaF16BF16Op`. Its `__post_init__` in `cute/nvgpu/warp/mma.py:100–118` accepts only f16/f16 with f16/f32 accumulation, or bf16/bf16 with f32 accumulation. | Actual selected-op gate is narrower than entry width checks. There is no connected FP8 MMA branch. C output type remains a separate copy/epilogue capability obligation, not supplied by the A/B op constructor. |
| `_compute_grid`:1121–1147, kernel:651–657,945–946,1005–1006 | `StaticPersistentTileScheduler`, cluster1, launch bounded by host active-cluster query. Both compute and DMA roles advance their deterministic work index. | Conditional old `persistent_tile_scheduler@1.fetch=static_stride`, `broadcast=none`; old `program` supplies blockIdx+j*gridDim ownership. Actual coordinate decoding and host grid bounds are the next source replay. This is not CLC cancellation. |
| Mainloop:726–837 | Clear accumulator per item, reset local K count only, preload shared fragments, process K tiles and release shared stages after register capture. Hoisted final MMA remains unconditional even when logical K count is0. | `physical_storage@1.pre_lifetime` supplies last shared read before next write; final arithmetic can outlive that read. Positive K and native fragment lifetime remain required. Zero fill alone does not establish an empty-K implementation. |
| Output:670–707,875–949 | Output buffer index and TMA-store object span the work loop. For each subtile, use current slot, publish writes, commit/acquire, synchronize all256 math lanes, increment slot modulo epi stages. After each work tile warp0 drains TMA source reads and all math lanes synchronize. | Exact old output rule's monotonic slot generation, read-retirement and tail obligations. This source has explicit tail and a post-acquire math barrier. An S2G source-read wait still does not authorize observing destination writes. Setup duplicates the initial store-object/buffer assignments; both copies precede work and are not two per-item rings. |
| Stage policy:1016–1056, setup:290–317 | Estimated AB stages=floor((capacity−2048−epi_bytes)/AB_bytes) at occupancy1; caller chooses epi stages. Setup exits the process successfully only if AB stages equal0. | Resource estimate omits exact aligned struct sizing. Negative stage results are not caught by this equality check. Since CLI allows any positive epi depth, sufficiently large output allocation makes the numerator negative. Native allocation/capacity and a positive-stage guard remain required; do not interpret exit0 as a completed kernel. |
| Capacity query:238, `cutlass/memory/smem.py:118–135` | No explicit architecture passed; capacity comes from the active DSL architecture enum and capacity map. | Target-dependent source binding, not an assumed SM90/SM120 constant or an observed device budget. Native register/shared allocations remain unmeasured. |

The hardest tentative reuse is matching source output reuse to the old rule's
`program`: all writers wait until the store warp's post-commit acquire has retired
enough source reads, and per-tile tail precedes the shared array's next work item.
The old rule supplies those edges; it does not supply native StMatrix/TMA layout
compatibility for every parsed C dtype. That remains for the layout correction.

## Checks and recovered scope

[Check](checks/geforce_dense.py) executes the retained argument parser at80 cases
(21 accepted), actual A/B width guards and selected-op constructor validation at343
dtype triples (77 pass width checks, only3 pass the selected op),126 stage-budget
cases and448 abstract output-ring domains. Seventy bounded budget cases produce
negative stages;17 produce zero. These budgets test the arithmetic/guard boundary,
not a measured shared-memory capacity. Every source blob used by the check is
verified against the manifest. [Result](checks/geforce-dense-result.json) retains
the domains and seven-tile menu.

Use the source-default f16/f16/f32-accumulator/f16-output, tile128×128×64, cluster1
as the pending native implementation witness. It passes the actual op constructor.
Shared stages depend on the target capacity and real layout; this study does not
invent a fixed stage count. The existing old stream/output proofs do not establish
that native instance until their transfer and resource premises are bound.

Host `run`:1215–1330 uses matrix fixtures, converts input/output buffers, compares
an FP32 einsum after converting the reference to C dtype, with caller absolute
tolerance and relative tolerance1e−3. The actual `matrix` helper's initialization
and output fragment copy selection remain for the correction pass. No numerical
range proof follows from the helper's generic name or the source comment about
random values. No compile, GPU execution or timing was performed here.

No new Template is proposed in this slice. Next: static scheduler coordinates,
StMatrix/output type support, shared layouts/alignment and host fixture/reference
details, then reconcile this entry before advancing to the remaining DSL families.

## Scheduler, layout and host correction — source closure

The [layout/scheduler check](checks/geforce_dense_layouts.py) executes retained
scheduler `create`, parameter `get_grid_shape`, linear `advance_to_next_work` and
FastDivmod coordinate expressions on scalar divisors.144 positive domains cover
6600 uniquely owned CTA coordinates, including batch3, non-square grids and
active limits above/below work size. Actual launch is `(1,1,P)`,
P=min(Mtiles*Ntiles*L,active_clusters), and initial index is **blockIdx.z**.
Next index adds P. With selected swizzle1/M-first raster, decode is
`(q % Mtiles, (q // Mtiles) % Ntiles, q // (Mtiles*Ntiles))`.
The old static-stride rule's logical worker index maps to grid.z rather than
literal grid.x; its `grid` field binds P, with the old domain1..4096 retained.
All cooperative compute and DMA roles reconstruct the same work sequence without
shared broadcast. Zero work/zero active limit is excluded from this binding;
the helper can otherwise return grid.z0. Native FastDivmod representation is not
replaced by the scalar divisor check.

Actual epilogue helper returns `(64,32)` for16/32-bit output and `(64,64)` for8-bit
output across all seven CLI tiles.28 type/tile domains cover475136 CTA coordinates
exactly once. This corrects any inference that the earlier arbitrary64×64 stage
arithmetic tests were the selected16-bit epilogue. They test the stage helper's
input domain, not a compiled instance's stage count.

The store helper returns `StMatrix8x8x16bOp` for16-bit C and `CopyUniversalOp` for
other widths; eight width/major choices are checked symbolically. The entry's
separate geometric `copy_atom_C` still constructs the16-bit StMatrix operation
with **C's actual type**. Its Python operation constructor does not validate that
type width; `_make_trait` forwards it to `CopyAtomStsmType.get`. Therefore this
study neither declares all non16-bit C supported nor claims constructor rejection.
Typed StMatrix partitioning, register retile, shared store and TMA composition
remain a native obligation. This is distinct from the **verified** FP8 A/B
rejection in `MmaF16BF16Op.__post_init__`.

A/B shared layouts use Hopper helpers selected by actual major/width/tile K;
C uses `make_smem_layout_epi`. All three arrays align1024 bytes, with two8-byte
input barriers per stage. The source's arithmetic capacity estimate is not a
field-for-field layout size proof. Register requirements232/40, all384 launched
threads, target capacity and native shared cosizes must fit together. Keep the
negative-stage and successful-exit0 gaps explicit instead of substituting native
resource assurance for a passing CLI parse.

The host `matrix` helper actually initializes signed values−2/−1/0/1, including C;
it does not follow the caller comment that says A/B0..2 and C zero. Output C is
not loaded by the GEMM, so its initial random value is irrelevant to this store
contract. These values are exact in the admitted f16/bf16 inputs and do not test
general range/rounding. C-major matrix storage uses original shape(L,N,M) and
permutes(2,1,0); other major uses(L,M,N),permutes(1,2,0). Native alignment/strides
and TMA overhang bounds still need validation. No explicit host alignment guard
is added by this source entry.

Reference einsum is FP32 then converted to C dtype before comparison. That
reference precision is not an FP16-accumulator semantics proof. Cold-L2 generation
rotates complete A/B/C argument sets; workspace count comes from the byte estimate
and benchmark helper, not a cache-state observation. The module prints PASS after
`run`, but setup can exit0 early when stage count is0; no source print is counted
as a real run here.

[Result](checks/geforce-dense-layouts-result.json) retains the scheduler, epilogue,
store-choice and fixture-default checks. All functions and branches in this
single-entry family now have source dispositions. No new Template or old proof
domain change is needed. Native layout/target capacity, positive K/stages,
output observation, precision and target gating remain implementation obligations.
Next: Blackwell attention's12 files across FMHA, mixed-input FMHA, MLA and Mamba2
SSD, then other outstanding DSL/C++ families in the full inventory.
