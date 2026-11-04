# Native GEMM and shared epilogue continuation

This trace follows the remaining BF16 and SM90 per-column-scaled FP8 paths in
[the pinned source](inputs/sources.json), plus both shared SM100 store layouts and
all four public epilogue classes. It complements the grouped/scale traces. Native
configuration enumeration and final whole-library reconciliation remain queued.
No CUDA compilation or device execution was performed.

## Native arithmetic and empty groups

| Source decision | Existing rule / binding or gap | Source boundary |
|---|---|---|
| `sm90_bf16_gemm.cuh:46–60`: merge input stages for Normal/K-major/K-major and one128-thread math WG | Derived pipeline grouping; `multi_buffer_rotation@2` only within its stage/event domain | At least10 original stages; merge floor(stages/5), retain floor(stages/merge), enlarge K accordingly. This reduces wait overhead by source intent, not measured evidence. |
| `sm100_bf16_gemm.cuh:44–70`: analogous merged K64 tiles | Derived grouping and instruction selection | Normal NT only, at least8 stages, merge floor(stages/8). K-grouped never merges; its alignment must divide actual BLOCK_K. The replay covers128 SM100 merge domains without inventing an independent K knob. |
| SM90 BF16 `:151–280`: persistent scheduler, TMA full/empty, register WGMMA accumulation | `persistent_tile_scheduler@1`, `per_slot_handshake@1`, `warp_specialization@1`, `compute_instruction_selection@2` provisional instances | Producer and math skip zero-K groups. Full WGMMA waits precede operand release; distributed-barrier tail drains before teardown. Major-layout and grouped scheduler equations reuse earlier trace bindings. |
| SM100 BF16 `:174–239,255–349`: one/two-CTA UMMA, swapped operands, dynamic valid M | Same provisional rules plus runtime instruction extent | Input TMA and issuer each execute max(1,ceil(group_K/BLOCK_K)), including empty groups. First MMA clears the accumulator. Empty values are selected as zero in the epilogue before the final store; this avoids reading uninitialized TMEM. |
| SM90 1D1D `:180–218`: change K-group operand tensor maps in device code | Transfer descriptor lifetime; existing layout rules only partial | Wait outstanding descriptor use, replace address/dim/stride in shared descriptors, copy to global, release/acquire tensor-map proxy. These are distinct from operand data barriers; actual proxy/race validation remains open. |
| SM90 1D1D `:254–351`: per-row/per-column K128 scale promotion, FP32 TMA reduction | `block_scaled_partial_accumulation@1`: partial_k=gran_k_a=gran_k_b128,gran_m=gran_n1 | Reset raw per K128; each output selects its own B scale; final += sA*sB*raw. C is mandatory, output FP32. Existing rule domains already admit gran_n1;15 new exact-rational outputs replay that binding. |

SM90 K-group BF16's API explicitly requires C, CPU group lengths and no prefix layout
(`gemm.hpp:704–711`). The host copies a distinct C to D before launching. Therefore
skipping a zero-K group preserves C as required rather than leaving an uninitialized
direct result. SM90 per-column FP8 launchers likewise require C and FP32 D, use a
K128 scale menu and always reduce-add the promoted output. SM100 can directly output
zero for an empty K group or add zero to prior C. A source path that issues a dummy
initializing MMA is different from omitting all compute and consuming undefined TMEM.

BF16 stores round product accumulators before BF16 TMA reduce-add when C is present.
That differs from adding C in FP32 and rounding only once. The retained scalar witness
uses raw product1.00390625 and C0.00390625: product-round/add gives1, combined FP32
then BF16 gives1.0078125. This is a numerical contract boundary shared with the
source BF16 tests' looser accumulate tolerance; it is not a failed GPU comparison.

## Single epilogue class and two physical layouts

`csrc/apis/epilogue_class.hpp` registers Identity, Alpha, FP8CastWithClamp and
BF16StochasticRounding. `csrc/jit_kernels/impls/epilogue_class.hpp` resolves exactly
one choice: class, alpha shorthand and `(D,SFD)` shorthand are mutually exclusive.
Identity/Alpha expose a plain-store coefficient for cuBLAS dispatch. FP8 cast rejects
C and requires FP8 D; stochastic rounding rejects C and requires BF16 D. Stochastic
conversion requests an architecture-specific target for cvt.rs. Classes do not compose.

`epilogue/sm100_store_cd.cuh` reads whole N fragments per row/lane. Its store extent
must cover complete scale groups for FP8. `sm100_store_cd_swap_ab.cuh` instead reads
eight values per lane spanning two rows and four N positions, transposes them through
STSM and stores a16×128 FP8 tile with one warpgroup. The latter uses dynamic effective
M to bound stores. Both fence TMEM loads, release overlapped columns separately where
applicable, release the full accumulator as soon as all reads are in registers, then
convert/store. Shared memory has a separate two-stage TMA ring; accumulating output
uses reduce-add, otherwise plain stores, with3D descriptors for batched/K-group output.
These are storage/lifetime and operator-fusion instances, not permission to erase
rounding or merge their release events.

`operators.cuh:122–221` specifies FP8 output numerics: BF16-round accumulator values
and the clamp bound, count values beyond that BF16 bound, reduce amax, clamp amax,
produce K32 UE8M0 scales, clamp elements, multiply by BF16 reciprocal and pack FP8.
Thus "count clamped" refers to the rounded BF16 domain. A raw value1.00390625 exceeds
raw bound1.002 but both round to BF16 one, so its source-domain clamp count is zero.
NaN/overflow behavior and actual conversion tolerance are unexecuted obligations.

The ordinary layout reduces all32 columns locally for one row. Swap-AB keeps lanes
with equal lane%4 together through XOR4/8/16; two BF16 components track two rows.
Only lanes0..7 write row SFs, in row order0/2/4/6/1/3/5/7. Packed SFD flattens
(batch,N) scale groups: address bytes are
`4*((((batch*N+n)/32)/4)*sfd_stride + row) + ((batch*N+n)/32)%4`.
N128 alignment keeps a packed word inside a batch. The checks cover27 batch/row/N
domains and3024 distinct live SF bytes; this metadata layout does not imply a plain
reshape of the output payload allocation.

Optional clamp count uses per-stream persistent readiness storage initialized before
graph capture. CTA0 zeros the count and release-publishes gridid+1; other active CTAs
acquire it before local counting. Finalization reduces per-thread counts once per warp
and atomically adds to the result. Grid-id freshness, initialization participation,
count overflow, stream concurrency and release visibility remain device obligations;
MHC's fixed128-wide epoch protocol is not silently rebound to this protocol.

## Deterministic random bits versus stochastic rounding assurance

`StochasticRoundToBF16` derives16 random bits from each N-quartet's FP32 bit patterns
and logical quartet column. The ordinary layout owns all four values locally; swap-AB
sums weighted partial hashes across lanes spaced4 and8. Unsigned32 arithmetic wraps.
The hash finalizer chooses distinct16-bit halves and coefficients by quartet offset,
then supplies the packed random halves to cvt.rs. There is no mutable RNG state.

The replay independently tags all256 (row,column) positions of the swapped8×32 fragment
and recovers the same logical quartet hashes as the ordinary layout. This establishes
coordinate/hash consistency on the witness, not uniformity, unbiasedness, independence
from data, the device conversion instruction or a distribution guarantee. A reusable
stochastic-rounding rule would need those statistical/numerical premises; no new
candidate is proposed solely from the source class name.

[Checks](checks/native_epilogues.py) and [retained results](checks/native-epilogues-result.json)
cover the new per-column promotion binding,128 merged-stage geometries, fragment/hash
identity, SF bytes and rounding boundaries. All four epilogue classes and ordinary/
swap store helpers now have source-backed dispositions. Configuration heuristics,
remaining native host/build variants and final source coverage audit remain next.
