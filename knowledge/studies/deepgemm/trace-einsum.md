# Einsum views and combined batch/K reductions

Pinned source: [manifest](inputs/sources.json). This trace reuses the SM90/SM100 GEMM mainloop traces and accounts for `csrc/apis/einsum.hpp` and the distinct `sm{90,100}_bmk_bnk_mn` kernels. CUDA target paths are source-selected; no compiled/device result is claimed.

## Canonical BMM view

For `bhr, hdr->bhd`, take B2/H3/D4/R5 with distinct coordinate tags. Permute A from `[b, h, r]` to `[h, b, r]`, retain B `[h, d, r]`, and permute D `[b, h, d]` to `[h, b, d]`. The canonical operation is batch=h, M=b, N=d, K=r. Each output is the original sum over r. No tensor elements move: extents and strides are permuted together. This introduces [tensor_axis_permutation@1](delta/tensor_axis_permutation@1/template.yaml), with bindings `(axis0, axis1, axis2)=(1,0,2)` for A/D, identity for B.

SM100-only FP8 `bhd, hdr->bhr` additionally permutes B with `(0,2,1)`, producing batch=h, M=b, N=r, K=d. FP8 `bhd, bhr->hdr` permutes both operands with `(1,2,0)`, giving batch=h, M=d, N=r, K=b. D already has the canonical order. The check compares all outputs against independent original-expression sums, not merely matching shapes. All six rank-three permutations also preserve180 addresses on a non-contiguous asymmetric source descriptor.

SF metadata has its own dimensions/strides. For the first expression the source permutes SFA `(1,0,2)` and retains SFB. A pre-transformed SFA must be constructed in canonical `[h, b, packed_k]` coordinates and inverse-permuted as a view before calling the API. Flattening a two-dimensional transform then reshaping to `[b, h, packed_k]` changes physical layout; after the source's permutation its stride(-2) is wrong. The replay distinguishes these maps with concrete addresses. This is a composition obligation, not a reason to assume that a data permutation implies a correct SF permutation.

## Dispatch and output contracts

BF16 accepts four hard-coded expressions. The two output-batched forms may use cuBLASLt if deterministic mode is off, the backend is available and the epilogue exposes a plain alpha scaling. Otherwise they select native SM90/SM100 BF16 batched GEMM. `bhd, bhr->hdr` is routed to cuBLASLt directly. Arbitrary expressions are rejected; parsing/canonicalization of general einsum is not implemented.

The FP8 forms are also fixed: the first expression has SM90 and SM100 routes, the other two require architecture major10. `fp8_bmm` validates major layout/shape and transforms operand SFs before dispatch. The SM90 route passes transformed_sfb as data but derives its major-layout tag from original sfb (`csrc/apis/einsum.hpp:245–248`); when the input transform changes that stride interpretation, caller/descriptor consistency needs a source-pinned CUDA case. Keep this source pairing explicit rather than treating its layout validation as evidence that every transformed case works.

Only the first FP8 expression accepts an FP8 `(D, SFD)` output pair. It selects the dynamic FP8 epilogue, requires arch10, batch stride=N and N divisible by128. These constraints keep a four-byte scale word within one batch's output columns. The shape of SFD corresponds to D viewed as M rows and batch*N columns; it is not permuted along with D's batch/M axes. Plain BF16/FP32 output and optional C follow their separately checked contracts. The nonlinear/clamp/stochastic epilogues remain in the shared-epilogue inventory group.

## Reduction over both batch and K

BF16 `bmk, bnk->mn` sums over both batch and K. The FP32 path requires C to be the same tensor as D (pointer, size, stride). For BF16 D, the host creates a zero-initialized FP32 workspace, calls that accumulated path, then copies/casts to BF16 (`csrc/apis/einsum.hpp:22–39`). The source accesses C through `c->data_ptr()` on the FP32 branch without a preceding has_value assertion; the usable source contract therefore requires C present even though the public argument is optional. No substitute empty-C behavior is inferred.

Both native launchers choose fixed tiles128x128x64, require K divisible64 and M/N divisible64, and flatten the reduction into `S*(K/64)` units. `split_factor=ceil_div(units, max(num_sms/num_mn_tiles,1))` is the number of units per partial, not the number of partials. Grid size is `num_mn_tiles*ceil_div(units, split_factor)`. A block's local reduction unit s maps through `sk=(partial*split_factor+s)*64`, batch=floor(sk/K), innerK=sk%K. One partial can cross a batch boundary.

Concrete witness: S3, K192 gives9 units; choose split_factor4, yielding partial sizes4,4,1. The first partial spans all of batch0 and the first K64 of batch1; the last covers the last K64 of batch2. The replay covers all576 (batch, k) indices exactly once. The independent integer contraction remains unchanged under this flattening; floating-point reassociation is a separate numerical obligation.

SM90 accumulates BF16 WGMMA products into FP32 registers, then `atomicAdd(float2*)` writes each in-bounds result pair. Row and column guards clip residue tiles; M/N multiples64 ensure paired stores do not straddle an odd final element. SM100 accumulates BF16 UMMA products in tensor memory, then stages FP32 fragments and uses `SM90_TMA_REDUCE_ADD_2D` into D. Both need initialized D/workspace and admit atomic ordering differences. Humming's `first_writer_atomic_combine@1` has an initialization protocol absent here, so it is not bound: source initialization plus a plain atomic-combine rule remains a separate gap. `reduction_partition@1` covers the exact-once partition; `linearized_work_partition@1` is the nearest source-order abstraction but has to be checked against its actual variable domain before binding this combined reduction.

## Decisions

| Source and binding | Rule / field or gap | Evidence / outstanding obligation |
|---|---|---|
| `apis/einsum.hpp:260–296`: three FP8 expression mappings | `tensor_axis_permutation@1` axes102,021,120 | Original-expression and canonical-BMM sums agree on all outputs for distinct extents/tags. |
| Same file`:223–229`: SFD output constraints | Packed scale ownership in family glue | N%128 and batch stride=N; quantizing output is a separate numerical policy. |
| Same file`:80–91,114–126,152–153`: BF16 backend choice | Dispatch heuristic/capability boundary | Determinism and epilogue capabilities change the backend; no unsupported operation is guessed. |
| Same file`:245–248`: original-SFB major tag / transformed-SFB pointer | Open layout consistency case | Need a transformed gran_mn1 SFB with changed strides traced into exact SM90 SF address selection. |
| Same file`:22–39`: FP32 accumulated output and BF16 workspace | `physical_storage@1.level=gmem`; conversion/initialization glue | Empty C is not handled before dereference on this FP32 path. |
| `csrc/jit_kernels/impls/sm90_bmk_bnk_mn.hpp:19–36`, SM100 counterpart`:19–34` | Fixed tiling and derived split amount | Tile128/128/64; source formula counts units per partial. No measured optimality. |
| `impls/sm90_bmk_bnk_mn.cuh:90–121`, SM100 counterpart`:99–124` | `reduction_partition@1` exact cover; combined reduction decoding |576 indices covered in a boundary-crossing4/4/1 witness. |
| SM90 body`:135–175`, SM100 body`:217–257` | `compute_instruction_selection@1`, `staged_relayout@1` partial matches; initialized atomic combine gap | WGMMA→register atomics vs UMMA→TMA reduce. Atomic ordering and numeric tolerance remain open. |

[Replay evidence](checks/einsum-views-result.json) covers addresses, integer algebra and split-index coverage. Device execution and external cuBLASLt internals remain outside that evidence. This group is source-traced with explicit partial bindings, not admitted as a complete executable Family.
