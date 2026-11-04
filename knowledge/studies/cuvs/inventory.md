# cuVS NVIDIA design inventory

Status: extracted by distinct design; candidate admission and realization proofs remain separate. Source scope is the retained superproject. External solver implementations are boundaries, not attributed to local CUDA bodies.

Reconciled `cpp/CMakeLists.txt:1483-1575`, public source wrappers, JIT matrix/export entries and their concrete device bodies. Full retained tree listing: inputs/source-files.json. The [source decisions](trace-additional-designs.md) contain bindings, variant differences and last verified boundaries.

| Material design family | Concrete implementation route | Disposition |
|---|---|---|
| CAGRA search | Single/shared frontier, multi-CTA local/global visited scopes, multi-kernel global stages, persistent descriptor service | Split-scope claim and persistent service candidates; approximation/reset/filter policies explicit |
| Graph construction | NN-descent new/new and new/old SIMT/half/quantized joins; Vamana greedy-search producer and ratio pruning; CAGRA ranked detour/reverse/protected-edge merge | Three graph candidates with distinct ownership and pruning invariants |
| IVF flat/SQ/binary | Interleaved scan, query/probe term hoisting, Hamming coarse metric, local top-k vs materialized scores | Hoisted metric, exact code metric and merge candidates; original input/SQ/CUTile cosine policies kept distinct |
| PQ/VPQ | Residual/VQ training, shared/subspace codebook selection, bitfield producer, shared/global LUT and chunked score scan | Concrete provisional Faiss lookup reuse; certified partial filter only with nonnegative remaining terms and sound threshold |
| RaBitQ/BBQ | QR rotation, residual signs, factor/error producer; LUT/bitwise short scan, optional long refinement; packed/plane integer dots | Code-dot candidate; 1.9 error coefficient is not deterministic bound evidence. BBQ prototype trainer is CPU/test-only |
| ScaNN | AVQ centroid quadratic solve, BF16 residual-projection coordinate update, SOAR spill objective | Coordinate-update candidate; explicit dense-solve/computation composition and quality parameters |
| Dense pairwise/fused NN | CUDA shared tiles, SM80 CUTLASS double/half tiles, optional cuTile source128x128x32 | Concrete output/reduction tile bindings, pair combine and row-constant score candidates |
| Sparse pairwise/cross-component | Dense/hash cached-row COO segmented semiring, filtered pair argmin and mutual reachability comparator | Cached segmented join and pair-combine bindings; explicit-zero support and atomic segment premises |
| Ball-cover / epsilon | Landmark-radius warp ballot and list scan; kNN weighted heuristics | Fixed-radius bound filter with equality preserved; no exactness promotion of heuristic weighted pruning |
| Flat/balanced/hierarchical KMeans | Fixed-centroid batch sums, donor reseed, coarse/fine quota initialization, global refinement | Event staging, donor reseed and hierarchical budget candidates; empty-cluster semantics explicit |
| GMM | Stable per-warp responsibilities, full centered Gram, raw diagonal/tied moments and spherical average | Centered Gram candidate; raw moments retain different numeric contract and regularized denominators |
| Spectral/linkage/PCA | Laplacian/eigen/KMeans chain, cross-component repair/MST/canonical edge sort, PCA solver wrappers | Concrete local composition; RAFT/cuSOLVER internals delegated, not invented CUDA mechanisms |
| Multi-index/device/tiered | Composite stream-pool children, replicated query scheduling, sharded root/tree merging, ANN prefix/BF tail | Partition-topk merge candidate, identity/partial-batch/completion obligations |
| Dynamic batch service | Reservation, pointer publication, deadline close, GPU gather, search/scatter/event | Committed-batch candidate; timeout cannot cancel a committed missing pointer |
| Refine/all-neighbors | Query-candidate virtual IVF list; overlap cluster membership then local build/identity merge | Concrete reuse of scan/merge mechanisms with supplied-candidate scope |
| Standalone quantization | Binary threshold bit transpose, signed int8 lroundf endpoints, PQ producer/codebook map | Bit-transpose and affine round-away candidates; calibration is a separate policy |
| Statistics | Batched silhouette labeled atomic sums; trustworthiness inverse-rank lookup and penalty sum | Seed output_tiling@2/reduction_partition@1 with explicit counts, ties and initial-state contracts |
| Core/filter/serialization/export | Resource/JIT wrappers, bitset/Roaring filter payload, emitted specialization menus | Supporting operations; Roaring container implementation belongs to next library, no separate full study or synthetic runtime claim |

Specific open obligations: RaBitQ confidence-bound applicability; quantization/TF32 rounding boundaries; device lock/event/token memory order; inactive-lane and packed-tail lowering; upstream ANN quality and external solver bodies. Each is attached to a read design. No major family remains represented only by a directory listing.
