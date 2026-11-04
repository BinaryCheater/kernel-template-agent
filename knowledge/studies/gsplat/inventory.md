# NVIDIA design inventory

Reconciled public wrapper/build groups with retained `gsplat/cuda/csrc` device bodies, `include` numerical helpers and Config.h build gates. [Concrete bindings](trace-additional-designs.md) include source locations and boundaries; [initial binning trace](trace.md) retains key/range replay. CUDA C++, no fixed SM assumption. Build selectors 2DGS/3DGS/3DGUT/Adam/relocation/losses determine compiled modules; channel list1..513 is a compiled menu, not an optimizer continuum. Camera wrappers are opt-in test exposure.

| Distinct family | Representative design disposition |
|---|---|
| EWA fused/simple, packed3D/packed2D | Jacobian covariance map; predicate_stable_record_compaction@1 for two-pass packed records; output_tiling@2 and keyed gradient reduction |
| Unscented nonlinear projection | nonlinear_sigma_point_moments@1; covariance validity, rolling pose and nonlinear culling boundaries |
| Blur/antialias opacity | blurred_gaussian_opacity_compensation@1 with inactive-floor mass premise |
| Dense/sparse tile intersections, ellipse/box, LiDAR | counted_overlap_depth_binning@1; periodic seam merge prevents duplicate output, same count/emit predicates |
| Sparse requested pixels | occupancy_bit_rank_addressing@1; uniqueness premise and matching published map |
| Projected-conic serial compositing | Ordered recurrence, shared_memory_tiling@1 staging; reverse_transmittance_replay@1 with cap/skip/checkpoint agreement |
| 2D surfaces and screen filter | Distinct homogeneous intersection and min support; ordered_pair_distance_prefix_moments@1 for sorted-depth distortion; median threshold state |
| World-ray serial/parallel | Distinct anisotropic ray-distance evaluation; ordered_transmittance_batch_summary@1, crossing-batch replay, parallel checkpoint backward |
| Priming and launch ordering | monotone_prefix_bound_hint@1; round-major launch permutation changes execution order while storage stays tile-major |
| Count, first-K IDs, top-weight IDs, pixel-Gaussian occurrence output | Different callback contracts; common accepted-hit recurrence; top-weight set returned in depth order |
| SH/DC versus L1+, view geometry gradients | Explicit degree map; reduction_partition@1 coefficient ownership and degree-sized FP32 register accumulation; shared camera offsets |
| Quaternion covariance/precision and storage | output_tiling@2; symmetric_matrix_packing@1 upper6/full9; record_field_axis_storage@1 for checkpoint SOA; physical_storage@1 only capacity/lifetime |
| Camera and external distortion | Declared camera maps; bounded Newton/inverse polynomial policy, rolling shutter; nested triangular Horner computation |
| Masked Adam | selective_parameter_state_update@2 proposed clockless domain; inactive complete-state identity |
| Relocation/MCMC/losses | Fixed declared mathematical maps; opacity clamp and scale approximation, covariance-shaped supplied noise, fused branch VJP |
| Multi-device key ranges/collective wrappers | Disjoint key-domain slices and host total ordering; system atomics/accessibility premise; Torch transport delegated |
| Null/camera wrappers | Identity map / numerical helper exposure, no new optimization mechanism |

No major design group remains unread. Formal correctness of camera inversion, geometric culling, managed-memory coherence and numerical thresholds is outside source extraction; exact binding premises are recorded rather than inferred from filenames.
