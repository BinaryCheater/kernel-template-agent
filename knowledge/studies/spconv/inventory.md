# Design inventory

Boundary: Python public ops.py and csrc/sparse/all.py registrations, cross-checked against csrc/sparse/*.py, csrc/hash/core.py and csrc/utils/*.py. NVIDIA generated C++; no particular SM chosen or measured.

| Design group | Representative and variants | Disposition |
|---|---|---|
| General/transposed neighbor generation | indices.py:294-337, 450-500; sort/unique vs direct table, bounded output recompaction | New offset_indexed_sparse_incidence@1; direct-table atomic internals remain cumm boundary |
| Mask and submanifold generation | indices.py:600-650, 748-805, 807-940; split mask and multiword mask | Same incidence rule; output domain and symmetric reverse-offset mapping differ |
| Native gather/GEMM/scatter | ops.py:811 onward; gather.py:31,59; convops.py | Incidence consumer; output_tiling@2/reduction_partition@1 only for legal submaps, external cumm instruction mapping open |
| Implicit GEMM / backward / int8 | ops.py:1450,1667; inference.py | Incidence consumer; quantized epilogue and generated MMA bodies not deeply read, named gap |
| Max/average/global pooling | maxpool.py:41,76,120,159,211,262,303 | Indexed reductions; avg divisor and max equality gradient contract are separate obligations; backward average not generalized |
| Point-to-voxel | pointops.py:138-246,252-291 | Hash grouping then atomic bounded append. Saturation selects a schedule-dependent subset, not exact unbounded grouping |
| Hash, sorting, gather/scatter and box utilities | hash/core.py, sparse/all.py, utils/boxops.py | Supporting lookup, ordering and maps; cumm hash/Thrust implementations outside retained superproject; box suppression family unread |

Coverage: source-design pass with explicit gaps, not full implementation coverage. CPU implementations excluded. No reopening of dependent libraries.

## Reconciliation after deeper reading

The [additional trace](trace-additional-designs.md) supplies concrete dispositions for all retained NVIDIA design groups above. gather.py and boxops.py contain CPU implementations, so remove them from NVIDIA coverage. Sparse hash insertion and MMA/quantization bodies are external cumm, whose allowed dependency range is pinned by the superproject but exact source revision is not. Device-body claims stop at the consumed interface. Full pipeline/ISA extraction cannot be supplied from that unpinned dependency.
