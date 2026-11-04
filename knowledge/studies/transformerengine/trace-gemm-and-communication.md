# GEMM and collective chunk handoff

When a communicated matrix consists of independently usable chunks, overlap can
move from whole-operation concurrency to a chunk dependency schedule. Keep the
arithmetic partition, publication protocol and final stream join separate.
Paths below are relative to `transformer_engine/common/` in the retained source.

| Source and binding | Rule/choice | Boundary |
|---|---|---|
| `gemm/cublaslt_gemm.cu:104–145,329–814` | External cuBLASLt GEMM with selected orientation, code/scale representation and epilogue | Descriptor flags, pointer/stride choices, scale modes, alpha/beta, bias/GELU and accumulation are visible. Vendor MMA tiling/pipeline is not. Canonicalization may select a separately quantized columnwise operand, not simply relabel the same bytes. |
| `gemm/cublaslt_batched_gemm.cu:170–295` | Strided-batch descriptor and MXFP8 scale-buffer layout | One shared shape/stride contract differs from a ragged group. Source rejects unsupported epilogue auxiliary output here. Batch scale padding and input orientation must match vendor consumption. |
| `gemm/cublaslt_grouped_gemm.cu:283–353,1060–1146` and device setup | Grouped pointer/shape/scale descriptor preparation; `runtime_work_descriptors@1`, segmented storage/address bindings | Device setup workspace and cuBLAS scratch are separate. Homogeneous representation, group offsets and transposes are checked before the external grouped matmul. Feature/version gates differ for MXFP8, NVFP4, tensor FP8 and Hopper; none is a measured capability in this study. |
| `gemm/cutlass_grouped_gemm.cuh:57–143` | Existing grouped/persistent/TMA/warp-role choices through a local library adapter | Visible default is SM90 pointer-array ping-pong, tile 128×128×128, cluster 1×2×1, stage count carved around epilogue shared memory. Underlying external CUTLASS headers are not materialized or studied here. This adapter does not resume the user-paused CUTLASS study. |
| `comm_gemm_overlap/comm_gemm_overlap.cpp:444–504` | Bulk independent communication and compute streams with common start/final join | Source assumes communication input is already in registered userbuffer. This overlaps independent operations; a communication **launch** event alone does not make newly gathered input ready for a dependent GEMM. SM allocation/copy-engine choices are performance controls. |
| Same file `:666–765` | New [chunk_ready_collective_pipeline@1](delta/chunk_ready_collective_pipeline@1/template.yaml), `direction=scatter_output`, `handoff_mode=stream_event` | GEMM output i completes before reduce-scatter i. Later GEMM launches can run while previous chunks communicate. Workspace is partitioned per compute stream; final communication/compute events join back to main. Last communication chunk may use a larger SM allocation. |
| Same file `:550–611`; `gemm/cublaslt_gemm.cu:732–765` | Same new rule, device-counter publication instead of a separate GEMM launch per chunk | cuBLAS atomic-sync input/output descriptors bind chunk dimensions and producer/consumer direction. Pinned code requires CUDA 12.2..<13 and cuBLAS 12.2.5..<13 at build and runtime. The vendor's internal chunk completion implementation remains external. |
| Same overlap file `:1169–1223` | Same new rule, `direction=gather_input`, stream-event ring | Rank r computes the currently resident chunk `(P+r-i)%P`, receives `(P+r-i-1)%P`, then makes the next compute stream wait for receive completion. Output offsets retain original chunk ID, not arrival iteration. An aggregate variant exchanges two chunks per step; its group/shape contract differs. |
| Same file `:940–977` and atomic AG body | Gather-triggered atomic GEMM and explicit launch epoch | Counters reset before this operation; local chunk starts ready. Temporary output includes an extra chunk for the selected atomic schedule. Buffer lifetime and provider counter ordering remain binding obligations; this is not an ordinary one-slot shared-memory handshake. |
| `comm_gemm_overlap/userbuffers/userbuffers.cu:773–843,2142–2200,2577–2650` | Local native P2P/collective mechanics: peer pointers, operation IDs, copy completion, counters | FP8 reduce-scatter decodes into two-byte accumulation/output; the selected kernel uses a shared factor and rank-rotated read order. Require that scale/precision policy, power-of-two rank mapping and registered memory contract. Push-copy path system-fences before system-atomic peer signal; receive completion can publish the next GEMM counter. Timeout diagnostics are not a proof of progress or completion. |
| `comm_gemm/comm_gemm.cpp:162–290,570–628` | External cuBLASMp AG+GEMM, GEMM+RS or GEMM+AR | Source owns distributed matrix descriptors, local shard shape and registered workspace. NVFP4 coarse A/B scales can be combined into device alpha. Device collective/tile schedule is unavailable. This backend must not be described as the local userbuffer implementation. |
| `ep/ep_backend.cpp:345–386,537–637,705–729`; `ep/nccl_ep_provider.cpp` | External NCCL-EP dispatch/combine with cached routing handle and transport metadata | Configure expert/token capacity, alignment, zero-copy/overflow and comm SM budget. Dispatch routes MXFP8 scales with token payload; per-tensor FP8 is rejected. Handle memory/configuration owns the prepared route epoch; backward reuses matching routing. Provider dynamic loading/JIT is not a visible native dispatch kernel in this archive. |
| `nvshmem_api/nvshmem_waitkernel.cu:22–54` | Kernel wait, NVSHMEM stream wait or driver stream-value wait, then signal reset | EQ versus GEQ is equivalent only under the caller's binary single-epoch signal contract. Reset must not erase a later publication. Kernel occupancy and stream wait progress differ; no generic arbitrary-counter substitution is established. |

## One concrete schedule

At rank 1 of a four-rank all-gather ring, source chunk order is 1,0,3,2.
Compute first consumes local chunk 1 while communication obtains chunk 0. After
the receive event, compute consumes chunk 0 while the ring obtains chunk 3,
then chunk 2. Each compute writes its original chunk's output slot. Canonical
concatenation is recovered when all streams join; ordering by loop iteration
would permute the matrix.

For reduce-scatter, producer i and communication i have a dependency edge even
if producer i+1 has already started on another stream. A complete matrix event
is safe but removes this granularity; a launch-only event is insufficient.
The new rule exposes the shared boundary and leaves the native visibility,
registered-memory, counter-epoch and progress contracts explicit.

Existing `cross_iteration_software_pipelining@1` chooses softmax/MMA skew and
warpgroup wait depth; it cannot instantiate these cross-kernel/collective stages.
`per_slot_handshake@1` uses local full/empty transaction barriers, also a different
realization. Payload storage, vectorized access, group work and numerical partial
reduction still reuse their existing rules. No new checker or timing was needed
for the dependency/partition argument; native distributed execution is unclaimed.

Scale-layout preparation is also local: `swizzle/swizzle.cu:28–83` packs batch
scale coordinates into 128×4 swizzle tiles. For scale row 129, column 5 and two
X tiles, the destination is 1553; batch stride is added separately.
`swizzle/swizzle_block_scaling.cu:273–308` expands a shared 128-value (or 128×128)
block scale into the corresponding 32-value MXFP8 metadata groups. Replicating
an unchanged decoder scale preserves values without requantizing the data.
`power_of_two_scale_encoding@1` applies only when its positive normal power-of-two
FP32 domain holds; arbitrary FP32 scales cannot be treated as exact E8M0 merely
because this conversion kernel exists. Padding, representation and native scale
layout remain the consumer binding.
