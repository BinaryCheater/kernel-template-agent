# Inventory and dispositions

Cross-check: faiss/gpu/CMakeLists.txt:24-69 and GPU index wrappers versus impl/ and utils/ device files. All paths below relative to faiss/gpu/.

| Group | Representative / differences | Disposition |
|---|---|---|
| Warp/block selection | utils/WarpSelectFloat.cu:14-80, Select.cuh:190-305,450-540 | New collective_triggered_candidate_queue@1. k=1 direct reduction; queue menus are compiled choices, not measured optima |
| Dense L2/IP distances | impl/Distance.cu:71-112,190-350 | output_tiling@2 on query/database tiles; L2 = norms - 2 dot, query norm can move after selection because constant per query. Rounding/clamp separate |
| PQ lookup construction | impl/PQCodeDistances-inl.cuh:28-100 | Per-code thread, subquantizer block; residual double buffer. multi_buffer_rotation@2 conditional reuse requires full lifetime binding, not asserted by buffer names alone |
| IVF flat / scalar quantized / PQ scan | impl/IVFFlatScan.cu, IVFInterleaved.cu, PQScanMultiPassPrecomputed.cu; GpuScalarQuantizer.cuh | Approximate list restriction belongs to workload; decoded distance accumulation and list-local selection. Detailed interleaved layout/decoder variants unread |
| Binary distance | impl/BinaryDistance.cu | Hamming XOR/popcount family inventoried; packing and tails unread |
| Append / residual / index remap | impl/IVFAppend.cu, VectorResidual.cu, IndexUtils.cu | Supporting placement/maps; append concurrency and identifier layouts unread |
| ICM encoder | impl/IcmEncoder.cu | Separate iterative encoding design, unread |
| cuVS-backed indexes | impl/Cuvs*.cu, BinaryCuvsCagra.cu | External backend calls; owned next in queue, no Faiss-specific device claim |

Half/float and direction instantiations share selector organization; comparator/tie policy remains an explicit contract difference. Build includes optional backend paths; availability is not evidence of dispatch or performance.

## Concrete coverage reconciliation

The [additional trace](trace-additional-designs.md) supersedes unread dispositions above. It follows interleaved producer/consumer mapping, both PQ table paths, SQ4/6/8 numerical policy, shared binary tiling and short-vector specialization, preassigned unique append, and full ICM update/evaluate/retain loop. Primitive norm/broadcast/residual/remap kernels are elementwise or sum maps already used in the dense-distance composition; they introduce no independent storage ownership mechanism. GPU code generation and CMake registrations account for direction, type and capacity specializations; these share the reviewed mechanism subject to their numeric/alignment guards.
