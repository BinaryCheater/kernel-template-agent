# SM103 scale composition and output-store lifetime

Paths refer to the [retained tree](inputs/sources.json). `sm103.py` denotes
`examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/`
`sm103_dense_blockscaled_gemm_persistent.py`. This closes the source-level interfaces
left by the [window/cluster trace](trace-cluster-variants.md); final compiler
intrinsics and device conformance retain their own obligations.

| Source choice | Reuse or gap / proposed binding | Evidence |
|---|---|---|
| `sm103.py:1932–1961`: scale atom `((8,4,4),(vector,4))` with strides `((16,128,4),(0,1))` | Same logical metadata layout as the host's32×4 row factorization; no new Template | The first two row factors coalesce because128=8*16.24576 address relations compare SM103 atom and ordinary atom at vectors16/32. |
| `sm103.py:1962–2070`: shared SFA/SFB stages, N256 SFB half offset | Fixed instruction-layout binding, separate from quantization vector and stage count | One SF stage has12 groups per row. For N256 the second128-column half starts1536 bytes later, leaving each half's three512-byte K-group atoms together.73728 staged positions are unique and agree with host tags. |
| `sm103.py:1217–1273`: shared-to-TMEM copy view and MMA iterator view | Metadata owner is checked; physical TMEM layout is a target intrinsic boundary | Source creates both views at identical SFA/SFB pointers; iterator K factors step6 or3 for vector16/32. make_tmem_layout_sfa/sfb delegate to `_cute_nvgpu_ir`; no invented integer address map is substituted for that intrinsic. |
| `sm103.py:1748–1804`; `python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/helpers.py:558–623` | cross_buffer_operand_window@1 instance: C256,W96 | Wrapper passes shared pointer, layout, major mode and optional next pointer. Pointer must carry16B alignment and valid shared swizzle metadata. Selected SM103 route calls make_umma_smem_desc; the SM107-family branch calls a different intrinsic and is outside this binding. |
| `sm103.py:1050–1198`; `python/CuTeDSL/cutlass/pipeline/sm90.py:1104–1112,1350–1358` | Separate AB/SF ring state with persistent generation context | Participant reset clears only count, preserving slot/phase. AB consumes three handles per K768 tile; SF consumes four or two. Do not reset the physical ring with each logical output tile. |
| `sm103.py:2118–2241`: separate stage allocation | Resource constraints plus fixed source heuristic | Reserve1024 bytes and initial C stages; allocate as many AB stages as fit after reserving one SF stage, then spend residual budget on SF stages, then additional C stages. ACC stages1 for N256 else2; C stages start2 for TMA or0 for direct output. This is a heuristic, not a measured optimum. |
| `sm103.py:1698–1739`; `python/CuTeDSL/cutlass/utils/gemm/sm100.py:178–307` | New staged_async_output_store@1, stages=num_c_stage | TMA epilogue writes converted results to shared slots, fences the async proxy, synchronizes writers, issues stores, commits groups and waits before reuse. It uses accumulator state.index; overlapping_accum is disabled for this path. |
| Same kernel; shared epilogue helper `:308–529` | Direct reg→global copy with bounds and vectorization context | Copy width=min(pointer alignment bits, common contiguous layout bits,256). NO_ALLOCATE cache policy is attached. Predicate uses vector's first logical coordinate; a complete-vector boundary premise is required.200 packet domains and a partial-vector counterexample distinguish that premise. |

The retained pure-Python `python/pycute/` layout algebra is loaded directly from its
verified snapshot. No installation or CUDA/DSL import is involved. Its actual
coalesce/tiled_product functions establish the composed scale address relations;
source-equation tags independently encode the host storage order. The larger N256
SFB stride is a physical layout obligation, not a new scale-value policy.

For the frontend-valid FP4/vector32 M128/N128/K768 instance, a scale buffer spans
K384 and stores128*12=1536 live bytes for each operand. Its atom factors8×4×4
coalesce to32×4 without changing addresses. For N256 SFB, the two128-column halves
each retain1536 bytes. Shared SF stage offsets use the filtered nonbroadcast size;
zero-stride K replication does not allocate another value per operand element.

## Output read completion versus destination completion

The existing async_transfer@1 body and effects fix global→shared destination
readiness. It cannot represent this shared→global output path merely by binding
mechanism=controller. The new [staged_async_output_store@1](delta/staged_async_output_store@1/template.yaml)
and [ProofOutline](delta/staged_async_output_store@1/proof.yaml) introduce the
output shared ring and distinct source-read/destination-write events.

`python/CuTeDSL/cutlass/pipeline/helpers.py:740–775,805–815` implements
TmaStoreFence.wait as wait_group(S−1,read=True), and tail as wait_group(0,read=True).
`pipeline/sm90.py:749–822` delegates PipelineTmaStore commit/acquire/tail to this
object. Those waits protect source shared bytes. The
[PTX bulk-wait reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)
defines read-only waiting separately from destination-write completion. A caller's
actual output visibility dependency remains separate; the source comments saying
“store complete” do not strengthen the read-only wait's semantics.

After commit u, at most S−1 unread groups remain, so group u−S+1 has finished its
source read before the next iteration reuses its slot. A scalar event replay covers
264 S/stream domains with independently delayed global writes. Source values are
captured before overwrite and later complete to distinct output locations. A weaker
S1 wait permitting one unread group leaves generation0 live when generation1 tries
to overwrite slot0; that witness is retained. Final read drain permits shared
reclamation, and the Template separately requires actual write completion before an
observer reads output.

The caller advances the tile scheduler before invoking its TMA epilogue delegate.
Therefore num_tiles_executed starts at1 there; `c_buffer=(executed*subtiles+i)%S`
rotates the ring's starting slot. Eight ring domains confirm this is a constant
rotation of the same lifetime sequence, not a missing output subtile.

The direct epilogue's vector predicate is sufficient only when the valid contiguous
extent does not end inside the selected copy vector. The host witness assumes16B
alignment and has complete output packets. The retained replay checks that premise
rather than treating one first-coordinate predicate as an elementwise mask.
N256 direct output retains reversed subtile traversal and early TMEM release when
overlapping_accum is enabled; its fence and arrival dependencies remain required.
Both output paths convert to C dtype before epilogue_op.

## Source extraction handoff

The four block-scaled Python entries now have source-backed dispositions for
contracts, packing, operand/scale ownership, scheduling, staging and final output
interfaces. Descriptor/TMEM encoding, target instruction availability, device
visibility/progress, NaN/padding numeric policy and compiled lane maps remain
implementation/admission obligations or declared source gaps. They do not require
fabricating an executable lowering to extract the three proposed rules.

[Checks](checks/README.md) record the source equations and their boundaries. Continue
with the six Blackwell dense-GEMM Python entries identified in
[dsl-family-index.md](dsl-family-index.md); the rest of the whole-library inventory
remains open. Source availability and declaration counts are not semantic coverage.
