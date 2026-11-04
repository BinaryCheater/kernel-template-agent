# Block-scaled variants, cluster assignment and completion

Source paths refer to the [retained tree](inputs/sources.json). `bs/` below is
`examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/`. Its four files
are reconciled in [variant inventory](inputs/blockscaled-variants.json); exact AST
matches identify shared helpers, not equality of the four kernels.

## Variant boundaries

| Entry | Source-backed differences | Disposition |
|---|---|---|
| `bs/dense_blockscaled_gemm_persistent.py` | Independent A/B types, FP4/FP6/FP8, UNPACK container paths and scheduler swizzle/raster settings. Same-type FP4/FP8 is a common subset with the prefetch companion. | Mixed Float6E2M3FN × Float8E4M3FN frontend witness M128/N256/K128, vector32, tile128×128, cluster1×1.750 dtype cases and25 UNPACK pairs checked. Full UNPACK encoding and mixed-layout fragments remain target obligations. |
| `bs/dense_blockscaled_gemm_persistent_prefetch.py` | Same AB dtype FP4 or FP8, extra independent L2 hints, default static M-major schedule. | Previous traces cover frontend tail witness, hints, scale windows, overlap and epilogue. This is not the baseline plus only a hint change outside their common subset. |
| `bs/dense_blockscaled_gemm_persistent_amax.py` | Same AB dtype FP4 or FP8, pre-conversion FP32 absolute maximum, per-warp shared reduction, scalar global atomic max. No accumulator-overlap branch. | Non-NaN arithmetic and reset witness below; exact padded/NaN behavior and GPU reduction lanes remain open. |
| `bs/sm103_dense_blockscaled_gemm_persistent.py` | FP4 only, MMA K96 over macro K768, separate AB/SF TMA warps and ring stages, circular descriptors, optional direct/TMA epilogue. N128/256 only. | Frontend M128/N128/K768,vector32,FP16 output,cluster1×1,TMA-store witness checked. Eight descriptor pairs and separate SF owners recovered below. Detailed descriptor/scale physical layout, phase helpers and epilogue delegates remain next. |

The ordinary baseline's `needs_unpack_tma` (`:1869–1899`) selects unpack when A/B
widths differ or either is6-bit. Same-width FP4/FP8 can stay packed. Narrow input
codes are stored in one-byte containers on the unpack path; this is not a claim
that they become FP8 numeric values. `:2094–2157` checks contiguous bits divisible
by128, then additionally requires128 contiguous elements for subbyte UNPACK. For
MN-major subbyte operands it checks the corresponding MMA tiler extent after the
two-CTA split. Consequently a B FP6 N-major tile192 fails even when the full N256
input is aligned; the two-CTA N128 tiler also fails because its split extent64 is
not128-aligned. The comparison must retain these constraints rather than copy the
prefetch companion's simpler `128/width` test into the mixed-width path.

## Persistent cluster assignment and multicast

| Source choice | Existing rule / glue or gap | Evidence |
|---|---|---|
| `python/CuTeDSL/cutlass/utils/static_persistent_tile_scheduler.py:170–246,328–365,461–682`: clustered grid and static stride | `persistent_tile_scheduler@1` expresses static_stride/natural assignment for the single-CTA witness. Cooperative-group lifting is explicit glue: the source work unit is a cluster, not one physical CTA. |2916 integer scheduler domains cover grid caps, batch decode, both raster directions and swizzle1/2/4. A cluster's work indices are z,z+grid_z,...; each local CTA gets cluster_coord*cluster_shape+local_coord. |
| Same scheduler swizzles | Additional permutation/padding context, not a timing Finding | Source rounds the non-raster cluster dimension to a swizzle multiple; the check retains87948 padded CTA-tile visits rather than discarding them as invalid work. Actual TMA/epilogue bounds still govern writes. |
| `prefetch.py:342–354,855–877`; `python/CuTeDSL/cutlass/pipeline/sm100.py:100–161,291–325` | `cluster_multicast@1` repeated per A/B operand, shared count<=4. SFA/SFB retain separate target layouts. | With MMA group V, cluster layout is `(V,CM/V,CN,1)`. A fans across N, B across M. The empty-release destination image is their union, plus the V-peer image for two-CTA MMA.15 configurations and70 leader release masks checked as coordinate sets. |
| `prefetch.py:795–808`; same pipeline helper `:329–381` | `async_transfer@1` controller readiness plus group release context | Empty count is CN+CM/V−1, the number of leader groups whose A or B source chunk is shared with this CTA. Each leader's outgoing image has V*(CN+CM/V−1) CTAs; count and image cardinality are different quantities. |
| `prefetch.py:812–824,1380–1406,1490–1598` and pipeline helper `:643–765` | Accumulator producer/consumer completion | Leader commits UMMA completion to the paired CTA group. Default prefetch epilogue expects128*V thread arrivals. The amax variant expects4*V and elects one lane per epilogue warp for release (`amax.py:721–736,1435–1441`). Mixing these policies would change the arrival count. |

For CM4,CN2,V2, the logical layout has two M groups and two N groups. Each target
CTA needs three leader releases (two along its row plus two along its column minus
self). Each leader's union reaches six physical CTAs, including V peers. The source
helper delegates final mask bits to a CuTe layout-image intrinsic; the replay checks
the intended coordinate image and count, not compiled mask bits or barrier execution.

The scheduler's `grid_z=min(total_clusters,max_active_clusters)` is the number of
persistent workers; physical CTA launch count is `grid_z*CM*CN`. Binding that count
as the seed's grid would change the static stride, while binding only grid_z as a
physical launch cost would undercount CTAs. Keep this source group context explicit.
The seed's `natural`/L2 order menu also does not itself encode the baseline's arbitrary
cluster swizzle and raster tuple. No blanket equivalence or measured L2 benefit is
asserted. Mixed fallback-cluster mode in the shared scheduler is a separate open
library route; these four callers do not pass fallback shapes.

## Amax numerical and lifecycle contract

`amax.py:1337–1365` takes abs/max over FP32 accumulator registers before the output
cast and epilogue lambda. `:1400–1434` reduces within each epilogue warp, writes four
shared maxima, waits, and lets first-warp lane0 atomically update one scalar. The local hierarchy corresponds conditionally to `cross_lane_reduction@1` and
`smem_partial_combine@1` (topology=serial,combine_op=max,participants4,result_to=one_warp).
The native warp_redux_sync backend differs from the seed's shuffle butterfly and
would need its own Realisation/contract evidence before admission. The global scalar
identity-initialized atomic max is a gap in this pinned pool: the existing
`first_writer_atomic_combine@1` is specifically additive and initializes via a
contributing partial, while `ordered_global_combine@1` fixes contributor order.
Neither represents this source's preinitialized commutative max by changing a binding.

Host `:2198–2209` initializes the scalar to negative infinity. The kernel does not
reset it for a new input. `python/CuTeDSL/cutlass/cute/arch/nvvm_wrappers.py:3932–3990`
implements `atomic_fmax(sign_bit=False)` with **signed** Int32 atomic max on the
incoming FP32 bits. Since incoming abs maxima are nonnegative, their ordered bit
patterns compare above the negative-infinity initial bits. Unsigned max would keep
negative infinity. The scalar replay checks6174 value/order cases, and distinguishes
pre-conversion amax2049 from FP16 output maximum2048. A later input with maximum2
would retain an old maximum9 unless the caller resets the scalar. NaN sign/payload,
warp reduction policy and contributions of padded accumulator positions remain open;
no general NaN-preserving max identity is inferred.

## SM103 cross-buffer windows and independent metadata stream

`sm103.py:265–272,1828–1853` fixes macro K768 and instruction K96. Its AB shared
layout (`:1855–1959`) supplies three K256 producer chunks. The unrolled loop
`:1332–1628` selects32-element subblocks with current indices0,3,6,1,4,7,2,5 and
chunk owners0,0,0,1,1,1,2,2. Windows2 and5 cross into the next chunk. The descriptor
helper `:1748–1804` passes current and next source pointers into make_umma_smem_desc;
no concatenation copy is added. Last consumers for chunks0/1/2 are windows2/5/7,
followed by UMMA-aware releases. Actual completion remains a target obligation.

Scale transfer is independent: vector16 uses four K192 SF chunks per macro tile,
vector32 uses two K384 SF chunks. Each SF buffer contains12 K-group values per
row/column. `MmasPerSfBuffer=8/buffers` and iterator stride6 or3 (`:1292–1293`)
make each K96 MMA consume its correct six or three scales.1536 tagged K-element
relations check those owners. This is not the same ring extent as AB; conflating
these two streams would shift metadata at crossing instructions.

The new [cross_buffer_operand_window@1](delta/cross_buffer_operand_window@1/template.yaml)
and [ProofOutline](delta/cross_buffer_operand_window@1/proof.yaml) encode this
address-only choice with chunk_extent256 and workload window_extent96. The bounded
check enumerates C1..32,W1..C and20 windows, in addition to recovering all eight
source descriptor pairs and release positions. Backend descriptor layout/encoding,
SF physical permutation, target legality and asynchronous completion remain explicit
premises. The chosen source instance is frontend-valid; no compiled kernel or GPU
behavior is claimed.

Next: finish SM103 descriptor/SF physical layouts and direct/TMA epilogue delegates,
then reconcile the remaining CuTe DSL mechanism groups and C++ families. The whole
CUTLASS study remains active. [All retained checks](checks/README.md) are linked once
here; no additional GPU measurement or admission is part of this extraction slice.


[Physical-layout/output follow-up](trace-sm103-physical.md) now checks the atom/host
composition and shared SF stage extents, follows both epilogue delegates, and
proposes staged_async_output_store@1. Source-read completion is distinguished from
destination completion. The source-level block-scaled route is extracted; the next
library group is the six Blackwell dense-GEMM entries. Named intrinsic/device gaps
above remain explicit implementation obligations.
