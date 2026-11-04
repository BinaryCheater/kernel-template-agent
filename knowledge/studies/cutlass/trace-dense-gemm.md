# Blackwell dense GEMM: six source routes

Paths refer to the [retained tree](inputs/sources.json). `dg/` abbreviates
`examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/`. All six entry classes,
methods and source locations are mapped in [dense inventory](inputs/dense-gemm-variants.json).
[Reuse audit](reuse-audit.md) applies structural retrieval cues and the hardest
reuse check to both new candidates and older field bindings. End-to-end agent
replay experiments are deferred.

## Entry contracts and implementation differences

| Entry | Distinct choices | Source disposition |
|---|---|---|
| `dg/dense_gemm.py` |128-thread CTA; warp0 issues TMA and MMA; explicit prologue lookahead P=min(S−2,T), then rolling input loads. One logical output tile per launch executor. | `kernel:707–799` supplies the manual generation schedule. Existing async_transfer@1 and multi_buffer_rotation@2 describe input copies and bounded slot mapping; the actual target completion remains a source-bound obligation. |
| `dg/dense_gemm_software_pipeline.py` | Same mainloop producer/consumer role is expressed as one loop annotated prefetch_stages=S−2. | `kernel:707–755` gives logical load/wait/MMA/release; compiler software-pipeline scheduling is an implementation boundary, not established by a straight source loop or by the manual example alone. |
| `dg/dense_gemm_persistent.py` | Six warp roles, static persistent cluster stride, explicit swizzle/raster, independent A/B dtype arguments. | `__init__:244–309`, `_create_tiled_mma:310–320`, `kernel:559–1061` use the shared static scheduler and common direct/TMA epilogues. Seed warp_specialization@1 cannot exactly express one-warp MMA plus separate epilogue resource counts. |
| `dg/dense_gemm_persistent_prefetch.py` | Static persistent loop plus independently chosen L2 hint distance for A and B. | `__init__:275–346`, kernel prefetch blocks and both epilogues reuse cache_prefetch@1 and staged_async_output_store@1 within their guards. Descriptor hints remain separate from data hints. |
| `dg/dense_gemm_alpha_beta_persistent.py` | Seven warp roles; a separate C-load role and C pipeline; separate shared C and D buffers; affine arithmetic in epi_dtype before D conversion. | `kernel:1024–1307` is a C-input/D-output composition, not an arbitrary post-conversion lambda. Input-C readiness and output-D source-read lifetime belong to different pipelines. |
| `dg/dense_gemm_persistent_dynamic.py` | Seven warp roles, CLC ownership transfer, one shared response slot, optional cluster-Z K splitting and DSMEM output-shard reduction. | CLC and scatter-reduce are new choices below. The original CLC coordinate-query IR and undefined empty K partials are explicit implementation gaps. |

For a common source/frontend witness, use Float16 A/B, FP32 accumulation, Float16
output, K-major A/B, N-major output, tile128×128, one-CTA MMA and cluster1×1.
M128/N128/K256 has16B-aligned contiguous extents and full output tiles. The helper
`python/CuTeDSL/cutlass/utils/blackwell_helpers.py:1248–1259` chooses instruction
K16 for same-type FP16; four instruction blocks make each mainloop tile K64.
The source-predicate replay checks this geometry and363 dtype triples per entry,
not a compiled tensor-level implementation.

The common dtype predicates allow FP32 accumulation with FP16/BF16/TF32/FP8,
FP16 accumulation with FP16 or FP8, and Int32 accumulation with Int8/Uint8; output
menus depend on the accumulator. Persistent and dynamic callers accept distinct
A/B arguments, but `_make_trivial_tiled_mma_impl:1248–1293` narrows actual pairs:
FP16/BF16, TF32 and integer paths require matching A/B types; FP8 variants can use
the paired low-bit instruction family. A broad frontend membership test does not
establish mixed-pair atom support. The explicit integer architecture guard rejects
SM107; the checker binds SM100 rather than dropping that target condition.

Direct output rejects incomplete CTA tiles. Static/dynamic persistent direct output
also requires complete swizzle sections. The checker accepts M128/N128 and rejects
M120/N128 with direct output. TMA output bounds, integer conversions, zero dimensions
and final lane maps retain their own implementation premises. The same output
arithmetic is not inferred for every exposed dtype merely from membership checks.

## Manual versus compiler pipeline

The manual prologue loads0..P−1 with P=min(S−2,T). Iteration i loads i+P when that
index exists, then consumes i; generation u uses slot u%S.264 source-equation cases
with S2..9,T0..32 preserve exactly one producer and consumer per logical tile. The
S4/P2/beta0/late0/private0 binding is inside the pinned multi_buffer_rotation@2
scope; S9 is outside that old rule's declared range and is not a reused-rule
admission. Readiness and release use PipelineTmaUmma handles, so their concrete
UMMA completion behavior is not supplied by a slot inequality alone.

The software-pipeline loop requests compiler lookahead S−2. It retains logical
acquire/copy/wait/MMA/release order, but actual emitted overlap, prologue, predicates
and barrier ordering require that compiler path's realization. This source slice
records the request and the shared semantic interface without inventing its lowering.
The persistent kernels instead separate TMA, MMA and epilogue roles and maintain
pipeline slot/phase across output tiles. Fixed roles are source instances; the seed's
four-warp consumer-group formula is a granularity gap for these one-warp MMA designs.

## C-load and affine output

`alpha_beta.py:1254–1306` loads C subtiles through a TMA producer and protects each
shared C generation until its epilogue reader releases it. Separate shared D slots
hold completed output values through asynchronous store-source reads. There is no
observed beta==0 branch skipping C loads. At`:1189–1197`, the operation is
`epilogue_op(alpha.to(epi_dtype)*acc.to(epi_dtype)+beta.to(epi_dtype)*C.to(epi_dtype))`
followed by conversion to D. Source `can_implement:1831–1884` checks the C dtype
menu and C/D alignment; it is not a comprehensive independent D-type/alias contract.

FP32 accumulator2049 plus C−2048 gives D=1 under this affine-before-final-FP16-cast
policy; casting the accumulator to FP16 first gives0. With explicit IEEE multiply/add,
beta0 and NaN C still yield NaN. The source expression does not justify treating C
as unread or uninitialized when beta0. FMA contraction, dtype-specific conversion
and NaN tolerance remain numerical implementation questions, separate from the
exact integer-valued cast witness. The C-input and D-output pipelines reuse the
input-transfer and output-staging contracts separately; affine fusion itself stays
a named field/ordering gap in the pinned seed, not a new name-based reuse claim.

## CLC launch ownership and response lifetime

`dynamic.py:727–744,1009–1032` adds a scheduler warp in the first CTA of the physical
cluster. `num_clc_stage=1`; response size16 bytes. Each response has
32*(1+physical_cluster_size*6) reader arrivals: controller plus TMA/MMA/four epilogue
warps per CTA. A one-CTA group has224, a two-CTA group416. The full original grid is
launched, rounded to clusters/swizzle; this is not the static scheduler's capped
grid followed by grid-sized strides or a software atomic-counter queue.

`python/CuTeDSL/cutlass/utils/dynamic_persistent_tile_scheduler.py:253–290,505–578`
submits cancellation and translates successful launch identities into work. Shared
response completion precedes generic reads and a proxy fence protects reuse. The
scheduler's local issued-query counter advances only in its controller; the dynamic
output role tracks its own completed-tile count for shared output-ring rotation.

The [NVIDIA CLC guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)
requires checking cancellation success before retrieving coordinates and forbids a
new request after an observed failure. In pinned `cute/arch/clc.py:63–143`, three
coordinate-query IR operations are constructed unconditionally after the success
query. Guarded emitted lowering or a corrected wrapper is required before this is
a safe Realisation; no such emitted artifact or GPU result is present here.

[launch_cancellation_work_reuse@1](delta/launch_cancellation_work_reuse@1/template.yaml)
and its [ProofOutline](delta/launch_cancellation_work_reuse@1/proof.yaml) state the
correct ownership and failure protocol. N1..5 bounded state models preserve the
partition into pending, exclusively active and completed identities, including
failure while some work remains pending. Uncancelled work still dispatches natively
under the fairness premise. The new choice is absent from persistent_tile_scheduler@1's
static_stride/atomic_counter domain. It is a candidate with an unresolved source
backend binding, not a claim that the original wrapper has been repaired.

## Cluster split-K and output-shard mailboxes

`dynamic.py:843–853` assigns rank r the clamped interval
`[min(r*ceil(T/S),T), min((r+1)*ceil(T/S),T))`. Its physical cluster has Z=S;
the CLC scheduler's logical K extent remains1 while launch Z expands by S. The
kernel decodes batch as work_z//S and retains physical split_rank for K ownership.
The source restricts split-K to direct output, swizzle1, raster M, group size<=16
and fragment elements per thread divisible by S. The static predicate replay checks
a positive S2 geometry and rejects TMA/swizzle2/raster-N neighbors.

`kernel:1181–1369` divides each128-thread output fragment E into S equal shards.
Receiver r keeps positions r*(E/S)..(r+1)*(E/S)−1. Sender s!=r writes r's shard
into slot s−(s>r), with mailbox offset thread+128*element+128*(E/S)*sender_slot.
FULL expects `(S−1)*128*(E/S)*4` bytes, then r reduces its own partial first and
other senders in ascending compressed-slot order. Only r stores its shard. Every
reader finishes before the receiver signals senders' EMPTY acknowledgements; each
sender waits for S−1 acknowledgements before its next generation's scatter.

[scatter_reduce_output_shards@1](delta/scatter_reduce_output_shards@1/template.yaml)
and [ProofOutline](delta/scatter_reduce_output_shards@1/proof.yaml) capture that
choice for defined FP32 partials.153 incidence cases, S2/4/8,E8/16/32,T0..16,
verify1340416 distinct mailbox coordinates and one contribution from each sender.
These are ownership/byte relations, not GPU visibility or arithmetic validation.
Other accumulator types need their own rounding contract; local fragment maps and
remote copy ordering remain target premises.

69 tested K-partition domains contain an empty rank. For T5,S4, the ranges are
[0,2),[2,4),[4,5),[5,5). In `dynamic.py:1072–1113`, an empty MMA loop performs no
observed accumulator identity write, yet completion is published and the epilogue
reads that accumulator. Correctness therefore needs an identity initialization or
exclusion of empty ranges; even T>=S alone is insufficient. The candidate's
local_partials_defined guard rejects this source binding. A valid nonempty example
is K256/T4/S2, where each rank owns two K64 tiles. No new empty-partial backend path
is inferred from the generic sum identity.

## Handoff

The six dense Python entries now have source-backed mechanism dispositions, including
failure/empty-partial gaps. Exact CLC/DSMEM ordering, compiler pipeline lowering,
source fixes, target atom legality and numerical acceptance remain implementation
or admission obligations. [Checks](checks/README.md) retain their commands, witness
inputs and boundaries; [reuse audit](reuse-audit.md) maps the hardest claimed reuse
to old fields or a named gap. Continue with the two Blackwell block-scaled grouped
GEMM entries in the family index, then remaining DSL and C++ families.
