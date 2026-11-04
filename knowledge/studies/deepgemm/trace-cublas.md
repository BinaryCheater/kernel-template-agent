# cuBLASLt adapters and external implementation boundary

All paths refer to [the retained DeepGEMM source](inputs/sources.json). This trace
covers `csrc/jit_kernels/impls/smxx_cublaslt.hpp`, GEMM public wrappers at
`csrc/apis/gemm.hpp:723–832`, the three fixed BF16 einsum adapters and their dispatch.
The NVIDIA library's chosen device kernel is external; no cuBLASLt call was executed.

## Layout and numerical contract

The internal plain GEMM accepts logical lhs `[M,K]`, rhs `[N,K]` with independently
K- or MN-major storage. It creates column-major descriptors and swaps operands:
cuBLAS A is rhs, cuBLAS B is lhs and column-major output `[N,M]` shares row-major D.
K-major rhs selects transA=T with descriptor `[K,N]`; MN-major selects N with `[N,K]`.
K-major lhs selects transB=N with `[K,M]`; MN-major selects T with `[M,K]`.
Leading dimensions come from the source tensor's corresponding major stride, not a
product of logical extents. Batched calls retain each operand's own batch stride.
NT/NN/TN/TT public aliases only transpose the logical argument views before this adapter.

This is a reuse of layout/view reasoning and instruction/backend selection. It does
not require a new Template: `tensor_axis_permutation@1` can describe simple axis views,
while the swapped backend descriptor is a source Realisation boundary. No admission
of an external library algorithm follows from creating valid descriptors.

`call_cublaslt_api` uses plain FP32 compute for block-scaled operands, otherwise
FAST_TF32 compute; FP8 fast accumulation is disabled. It requests runtime's SM target,
per-stream workspace and one heuristic algorithm satisfying workspace/reduction-scheme
preferences. No result is an assertion failure; the source does not search/tune a
complete algorithm space. SM count target and workspace size are policy inputs, not
actual occupancy or measured performance. Normal success destroys preference, layouts
and operation descriptor. Errors before cleanup can leave descriptor resources live;
this is an unexecuted error-path obligation.

Beta is1 when accumulating and0 otherwise; C and D pointers passed to cuBLAS are both D.
`gemm::early_return` first copies a distinct C into D for nonzero K. For K0 it copies C
or zeros D and returns; empty M/N returns before further work. Source dtype/shape/major
validation must precede valid use; explicit cuBLAS wrappers are thinner than the native
BF16 API. The helper's alpha multiplies the product, not the copied C. Accumulator
rounding and FAST_TF32 behavior belong to the external implementation's numerical
contract, not the integer source replay.

## Registered and fixed-expression variants

| Entry / source path | Concrete mapping and reused choice | Disposition |
|---|---|---|
| `gemm.hpp:723–754`, four plain aliases | Logical view transpose then independent major descriptors; alpha1 in explicit wrappers | Four major combinations and padded batch strides checked; backend numerical/device behavior open. |
| `gemm.hpp:757–788`, NVFP4 NT | Both packed FP4 operands K-major; logical K and leading dimensions twice byte extents | K divisible32; SF bytes validated by extent. Descriptor/type branch traced; actual packing/scale interpretation remains external. |
| `gemm.hpp:796–809`, batched_syrk | D=A@A.T using full strided-batched GEMM; 2D input becomes one batch | Full matrix output, not triangular-only storage. Same generic layout mapping. |
| `gemm.hpp:812–832`, batched_symm | D=A@B, A symmetric by caller contract; B view transposed before GEMM adapter | Symmetry is not data-validated and the wrapper does not exploit triangular storage. |
| `einsum.hpp:60–90`, bhr,hdr→bhd | h batches; rhs column descriptor `[R,D]` transposed; lhs `[R,B]` | Output coordinates (d,b,h) follow original D strides; dispatch only plain/alpha-compatible epilogue and nondeterministic mode. |
| `einsum.hpp:93–124`, bhd,hdr→bhr | h batches; rhs `[R,D]` plain, lhs `[D,B]` | Same backend gate; no extra tensor copy is implied by descriptor batching. |
| `einsum.hpp:127–147`, bhd,bhr→hdr | h batches; reduce B, output column descriptor `[R,D]` | FP32 D; direct backend call even when deterministic runtime mode is set. Source TODO leaves that guarantee open. |

The NVFP4 path requests VEC16_UE4M3 scaling, distinct from native MXFP4 K32/UE8M0.
The API accepts byte/char SF storage with size
`ceil(MN/128)*ceil(K/64)*512`, documented as `[MNtile,Ktile,32,4,4]` bytes.
For M129,N257,K96 the operands need2048 and3072 SF bytes. After operand swap, rhs SF
must be cuBLAS A's scale pointer and lhs SF cuBLAS B's; the source does that explicitly.
Byte count alone does not validate live SF values or the physical scale permutation,
and the existing exponent encoding rule cannot be substituted for this UE4M3 recipe.

The generic BF16 API chooses this backend only when deterministic mode is off,
cuBLASLt is available and the epilogue exposes a plain-store alpha. Native paths cover
other epilogues and deterministic dispatch. Explicit cuBLAS entry points invoke the
backend by request. `tests/test_bf16.py:217–319` tests normal alias/alpha behavior and
2D/3D SYRK/SYMM against reference products, but does not expose the internal chosen
algorithm. Their performance measurements and tolerances were not rerun or imported.

[Source-equation checks](checks/cublas_layouts.py) and [results](checks/cublas-layouts-result.json)
cover64 major/batch geometries with896 integer outputs and630 asymmetric einsum
contraction coordinates. Each descriptor load matches a separately tagged logical
operand. Checks also retain asymmetric NVFP4 SF extents and source pointer-swap markers.
No CUDA scale decoder, external algorithm or floating-point execution is represented.
Final disposition: all adapter branches accounted for; chosen device implementation,
ISA, numerical tolerance, error cleanup and deterministic guarantees remain external
implementation obligations. Continue remaining native GEMM/configuration/epilogues.
