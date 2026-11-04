# SM120 scale/TMA layouts and host contract

When a low-float MMA consumes separate scale metadata, keep three mappings
explicit: logical scale group, global blocked storage and shared/register operand
layout. An identical logical value need not have the same flat address in all
three. Reuse the existing transfer/operand rules only after binding those maps;
a scalar coordinate check does not establish native TMA or LDSM lowering.

This completes source extraction of the three-file GeForce blockscaled family,
alongside [dispatch](trace-geforce-dispatch.md) and
[schedules](trace-geforce-schedules.md). The [inventory](inputs/geforce-blockscaled-variants.json)
records both standalone kernels and their shared helper. Paths below refer to the
[pinned CUTLASS tree](inputs/sources.json). Named source/resource/numeric gaps stay
open; no native capability is promoted by this source-study closure.

## Selected witness and maps

Use mixed FP4 A/FP8 E4M3 B, E8M0 SF/vector32, FP32 accumulation, FP16 output,
M128/N256/K640/L1, tile128×128×128, epilogue64×32, cluster1. Both CLI dtype/tile
menus accept this binding. Cooperative uses eight MMA warps on each tile;
pingpong uses four warps per alternating tile. Source-derived stages require the
actual shared-memory capacity/layout calculation; the earlier S3 stream example
was logical rather than a reported native allocation.

| Decision and source | Actual mapping | Owner and remaining obligation |
|---|---|---|
| `_setup_attributes` and `_make_smem_layouts` in both kernels | Same-type FP4 uses four-bit shared element allocation. Mixed FP4 uses Int8 shared allocation/TMA internal type on that side; FP8 stays native. A/B layouts use the Hopper atom selector with target tile/major mode and1024-byte buffer alignment. | Source implementation binding. Allocation element width, logical tensor width and TMA internal width have separate roles; they cannot be substituted into one byte estimate. Native reinterpretation, lane mapping and actual copied-byte agreement remain required. |
| Cooperative:320–362,1421–1458; pingpong equivalents | Data TMA uses the side-specific internal type; SFA/SFB TMA uses Int16 as an opaque internal type. Cluster is1, so actual calls select non-multicast G2S; C uses S2G. | async_transfer@1 global→shared direction applies conditionally. Int16 scale transfer is storage interpretation, not floating scale conversion. Multicast helper availability is not a selected multicast instance. |
| `utils/blockscaled_layout.py:269–438` | MN atom(32,4), strides(16,4); K atom(sf_vec,mma_K/sf_vec), strides(0,1). Remaining K and MN axes choose explicit block strides; each stage appends the zero-filtered layout extent. | Scale repeats within its logical sf group by stride0. Scalar metadata yields one address for each(row,k//sf), without collision between distinct scale groups. Physical stage strides depend on filtered layout support; native copy/fragment composition remains open. |
| `utils/blackwell_helpers.py:1663–1704` | M permutation capped128; N shape(8,2,2), strides(1,16,8); K permutation32 for FP8/mixed or64 for same-type FP4. | Exact source operand-coordinate permutation, not output-tile rasterization. Its32 N positions form a bijection with the middle groups exchanged. Neither an old three-axis permutation label nor a cache-order knob alone supplies this nested map. |
| Same helper:2638–2712 | SFA TV broadcasts over the warp-N direction; SFB TV broadcasts over warp-M. Composition with thrfrg_SFA/SFB and the inverse thread layout selects each lane's scales. | Concrete native layout binding. Repeating a scale via stride0 does not permit dropping its distinct row/group identity. Scalar shared/global maps do not prove these composed native lane maps. |
| Cooperative stage policy:1188–1245; pingpong:1252–1309 | Output stages=min(number of epilogue subtiles,4). AB stages derive from capacity, occupancy, helper reservations, output bytes and AB/SF bytes. Mixed byte allocation participates in that policy. | Derived coupled budget, not independent stage choices. Both output and AB stages must be legal/positive. Old multi_buffer_rotation@2 stops at S8; a larger source-derived count needs separate support. No allocation or occupancy measurement is inferred. |
| Cooperative host:1663–1745; pingpong corresponding helper | Logical reference scales shape(L,MN,ceil(K/sf)). Blocked allocation shape(L,ceil(MN/128),ceil(scaleK/4),32,4,4), view permutation(3,4,1,5,2,0); converter copies logical MKL coordinates into grouped blocked axes. | Host and shared maps are separately injective but have different outer-block pitches. TMA must preserve logical coordinates across them. The CPU CuTe converter and native format conversion are not executed in this study. |
| Host scale initialization | Valid reference scales use integer random values1 or2; padded float32 blocked storage is initialized to0 before conversion into E4M3/E8M0. Expanded reference repeats each scale sf times and truncates to original K. | Test fixture, not a quantizer. Valid scales are exactly representable in selected formats. Encoded E8M0 padding and data/metadata tail neutrality remain native numeric obligations; requested float32 zero fill does not prove consumed encoded-zero behavior. |
| `cutlass/torch.py:218–270` | Default signed matrix creation uses random integer range[-2,2), hence values-2,-1,0,1. These are converted to narrow input types. | Exactly representable test inputs hide general input quantization error. The reference multiplies raw fixture inputs by expanded scales then performs einsum; it does not establish accuracy for arbitrary floating source data. |
| Public `run` wrappers:1529–1571 /1593–1635 | Wrapper always passes Float32 accumulation even if callers include another acc_dtype through kwargs. Direct run_bs/CLI depend on native constructor enforcement. Layout CLI restricts A/B K-major and C N-major. | Public wrapper contract and source menu. Direct major-mode paths or a different accumulator type are not independently validated by the parser. Alignment precheck occurs before string normalization; witness uses dtype objects. |
| Host comparison:1784–1840 and pingpong equivalents | FP32/FP16/BF16 compare against float reference; FP8/FP4 outputs first round-trip the reference through the output type. atol is a user option and rtol1e-2. | These are reference policies, not a measured numeric contract or verified output capability. Other parseable output types receive no listed comparison branch. General cast/saturation and native output-copy legality remain open. |
| Benchmark workspace paths | Allocate fresh A/B/C and scale tensors; benchmark helper selects hot/cold workspace behavior. | Benchmark infrastructure is traced only. No timing, throughput or cold-L2 performance claim is retained. |

## Coordinate evidence

For local shared scales, the source nested strides reduce to
`512*((scale_col//4)*(MN//128)+row//128)+16*(row%32)+4*((row//32)%4)+scale_col%4`.
Every data K coordinate inside one sf group aliases its scale; distinct
(row,scale_col) pairs do not alias. Host blocked strides instead put the outer MN
block before the outer K-scale block in the original contiguous allocation.
Its permuted view preserves the same logical(row,scale_col,batch) identities.
Copying between them requires a logical-coordinate TMA binding, not equal flat
indices. Native tiling remains an implementation obligation.

[Layout checks](checks/geforce-layouts-result.json) execute the scalar metadata
prefixes of both retained SM120 scale helpers:24 domains and884736 coordinates.
Seventy-two host-layout domains check161568 distinct scale coordinates, including
partial MN/K groups and multiple batches. Helper domains outside a CLI tile menu
are layout checks only, not new accepted kernel configurations. The N permutation
covers all32 positions exactly once.

## Order storage clarification and final disposition

The previous two-entry order-array declaration did not itself account for four
PipelineOrder indices. The [new scalar backing model](inputs/geforce-order-padding.json)
adds the following source-layout argument: mainloop barrier bytes occupy16*S,
the declared order range adds16, and the following sA field aligns to1024. For
S1..16, the required four-barrier end16*S+32 remains inside that alignment gap.
Thus physical padding can back those extra addresses under the stated field-order
and alignment assumptions. The declaration mismatch was **not** evidence of native
out-of-bounds. Actual compiled struct offsets and alias/lifetime behavior remain
unverified; no blanket source failure or native fix is claimed.

All source decisions now have dispositions across the three traces. New candidates
from this family are operand_bitfield_placement@1 and
interleaved_consumer_streams@1. Cooperative final output-read retirement for S>1,
K0 hoisted MMA, missing reverse-mixed B alignment, general encoded padding,
constructor/format support, native register/TMA maps and order-storage backing
remain explicit source/implementation obligations. No GPU execution, CUDA/DSL
compilation, numerical GEMM result, Realisation, ProofSummary or Store admission
occurred. Continue with the two Blackwell GeForce convolution files.
