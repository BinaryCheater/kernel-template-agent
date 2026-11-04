# Legacy Triton, auxiliary TileLang and IDE indexing

Paths below refer to [the retained revision](inputs/sources.json). All four legacy
device functions, their NT/NN/TN wrappers, shared tuning menu, TileLang activation
helper and shape/scale helpers were read. No Triton/TileLang compiler or CUDA kernel
was executed. [Replays](checks/legacy_auxiliary.py) and [results](checks/legacy-auxiliary-result.json)
cover bounded source equations and execute the actual pure SF-shape functions extracted
from the retained source AST; they do not import or execute GPU dependencies.

## Legacy grouped GEMMs

`deep_gemm/legacy/__init__.py` exports ordinary M-grouped BF16, A-gather M-grouped,
A-gather K-grouped and B-gather K-grouped functions. `tune_options.py:10–28` supplies
nine source configurations, including a duplicate M32/N64/K128/stages3 entry. Filters
use estimated BF16 shared bytes `(BM+BN)*BK*2*stages <=166912` and BM/BK no larger
than the configured contiguous-layout alignment. This is an Ampere-oriented tuning
menu, not an architecture-enforcing host gate or measured winner. Autotune uses an
empty explicit key; generated specialization/cache behavior belongs to the external
Triton implementation. Runtime alignment defaults128 and its setter is unvalidated.

| Family / source choice | Existing rule binding or gap | Source domain and evidence |
|---|---|---|
| All four files: GROUP_SIZE_M8 raster, smaller last M group | `block_rasterization@1` provisional comparison | 363 independent M/N tile-count domains produce each tile exactly once. The shortened group's order can rotate with global pid; coverage is the checked property. |
| `m_grouped_gemm.py:34–58`: one expert ID at tile's first row; negative ID zeros the tile | Group ownership and `boundary_predication@1` partial | IDs must be homogeneous over each selected BM tile and valid IDs below B. Host checks shape/type/contiguity, not ID values. M has no row mask; safe bounds require BM divides M. N/K tails do have masks. |
| `a_fused_m_grouped_gemm.py:44–63`: gather A rows once, reuse across K iterations | `indexed_gather@1`, granularity=row,index_prefetch=registers; index_reuse depends on K/BK | Negative row IDs zero A; nonnegative IDs need upper-bound validation. Seed precondition requires all indices valid, so sentinel rows require an explicit predicated extension rather than a silently complete binding. Gather reuse above64 also exceeds that rule's domain. |
| Both K-grouped files: group start/end, complete BK loads, A- or B-side gather | `reduction_partition@1` partial and `indexed_gather@1` row instance | A-fused reads A[index[k],m]*B[k,n]; B-fused reads A[k,m]*B[index[k],n]. Negative gathered rows zero products. Source-equation replay matches independent integer contractions for120 outputs. Arbitrary per-group tails are not masked. |
| Both K files: ACC flag and autotune restore_value=['d_ptr'] | Accumulator initialization/update ownership | ACC adds existing D; empty group preserves D when accumulating and writes zero otherwise. Autotune restores D between trials so tuning must not repeatedly accumulate into caller state. No atomic combination is involved. |
| M NT/NN wrappers and K TN wrappers | Layout adapter, not new arithmetic | NT accepts B contiguous `[B,N,K]` or its transpose-contiguous strides; NN delegates through b.mT. K inputs are contiguous `[K,M]`/`[K,N]`. D in K wrappers is not constrained to BF16; accumulator converts to D's element type. |

The K loop visits `range(start,end,BK)` but loads every `base+arange(BK)` index,
without a `k<end` predicate. A five-element group with BK4 therefore visits0..7.
Whole-group BK divisibility, allocated source/index storage and valid padding are
separate premises; a global K multiple does not establish every group's boundary.
`tests/test_legacy.py:47–55` explicitly restricts tested K cases to alignment128,
non-prefix layout, accumulate=True and FP32 output for this reason. The tests also
zero invalid M rows after the call, so their final comparison alone does not establish
a device negative-tile store contract. No reported source benchmark is reused as a
KTA measurement.

The M tuning filter uses BM<=alignment, not divisibility. With a changed alignment96,
BM64 can pass this filter while M96 needs a128-row launch extent with no M predicate.
Default alignment128 makes the listed power-of-two tile sizes divide aligned M, but
that default is a necessary instance premise, not a guarantee for arbitrary runtime
settings. All arithmetic grouping, gathers and boundary choices match existing rules
conditionally; no additional Template is proposed for these legacy paths.

## Auxiliary weighted SwiGLU and scale production

`third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py` is a comparison/baseline helper,
imported by `tests/test_mega_moe.py` only when external DeepEP/TileLang are available.
Its input is BF16 concatenated gate/up, optional FP32 token weights, optional available
count and clamp. Gate is upper-clamped only, up symmetrically clamped; FP32 weighted
SwiGLU then uses a per-token/per-channel-group absolute maximum floored at1e-4.
`utils.py:34–46` either uses FP32 scale=amax/448 and inverse448/amax, or rounds scale
up to a power of two with exponent/mantissa extraction and constructs the reciprocal
from the negated exponent. UE8M0 requests store that exponent code; valid use needs
round_scale=True. `power_of_two_scale_encoding@1` covers only encoding an already
defined scale in its admitted range, not this floating-point amax/division/rounding
policy. Numerical overflow/underflow and conversion admission remain separate.

Persistent geometry derives hidden blocks as H/gcd(H,16384), per-block hidden width
from that quotient, token rows per block as1024/(block_hidden/16), and uses1024 threads.
CTA count is divided by hidden blocks, excess CTAs return, and token progression is
strided over the remaining CTAs. The per-thread loop bound corresponds algebraically
to token<available via thread_idx/layout_h. Generated predicate/reduction lowering is
an unexecuted compiler boundary, especially for partial token groups.

SF storage has three declared layouts: FP32 row-major, FP32 column-major with token
extent aligned4, or packed exponent bytes `[scale_word,aligned_tokens*4]` that become
int32 before transpose. The wrapper always returns `y_sf.T[:num_tokens]`. Executing
its actual shape helper for T7,H256,C32,FP32 row-major allocates `[7,8]`, but that return
expression produces `[7,7]`. The row-major branch is therefore not a consistent public
SF view. Packed UE8M0 also requires column-major backing; the wrapper does not enforce
the comment's stated requirement. The source MegaMoE callers all choose column-major;
BF16 baseline requests unrounded FP32 scales and BF16 side output, FP8 baseline requests
rounded UE8M0 K32 scales. These concrete caller branches avoid the row-major shape gap.

Global H divisibility by the scale group does not prove per-hidden-block divisibility.
For H49152,C96, three hidden blocks give block_hidden16384, which is not divisible96.
The fragment shape truncates to floor(block_hidden/C) groups despite j visiting every
hidden element. This is a source specialization-domain counterexample; no compile
failure or device outcome is claimed. Optional BF16 output is a side output: the
source still builds FP8 and SF fragments, so it cannot erase those geometry obligations.
No new Template is extracted from unsupported flag/shape combinations.

## Indexing target disposition

`CMakeLists.txt:1,33–34` explicitly says the static CUDA target is for IDE indexing.
`csrc/indexing/main.cu` contains implementation-header includes and `main(){return 0;}`;
it registers no data indexing operator or launch. Setup builds the Python C++ entry
and packages JIT include files separately. The inventory's former label "Indexing
extension" was misleading and is corrected to "IDE indexing target". Its includes
are references to already inventoried device families, not another runtime family.

Source-study disposition: the four legacy families and wrappers, auxiliary computation
and layout/scale flags, and IDE target have accounted branches. Compiler lowering,
exact numerical tolerance, GPU bounds and the retained unsupported-domain cases remain
implementation obligations. Continue with cuBLASLt adapters and the remaining native
GEMM/configuration/shared epilogue inventory.
