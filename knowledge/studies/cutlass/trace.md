# CUTLASS SM100 block-scaled persistent prefetch — first slice

Paths refer to the [pinned retained source](inputs/sources.json). This slice compares
`examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/`
`dense_blockscaled_gemm_persistent.py` with the neighboring
`dense_blockscaled_gemm_persistent_prefetch.py` (below, `prefetch.py`). It follows
producer index equations, not a compiled witness of the entire GEMM.

The two implementations have different capability menus: the ordinary baseline also
supports separate A/B dtypes, FP6/UNPACK and scheduler swizzle/raster settings. The
cache-hint comparison applies to their common same-type FP4/FP8 subset; it is not
an equivalence claim over their full domains. [Variant trace](trace-cluster-variants.md)
reconciles those differences.

Both are SM100 CuTe DSL batched dense block-scaled GEMM implementations. A is M×K×L,
B is N×K×L, C is M×N×L; SFA/SFB supply groups of K elements. Accumulators are FP32.
Exact supported dtype/layout/shape menus remain the next caller-contract trace.
The prefetch constructor fixes six warps: four epilogue, one MMA, one TMA. M tiler256
selects a two-CTA MMA group, otherwise one CTA (`prefetch.py:200–270`).

## Observed choices

| Source and concrete binding | Existing rule or gap | Evidence / remaining boundary |
|---|---|---|
| `prefetch.py:432–440`: None maps D to AB stage count S;0 disables;positive explicit D enables | Separate cache-hint distance choice; candidate gap, not `multi_buffer_rotation@2.P` | Source permits D>S; a hint has no shared destination slot. Negative values also disable under `>0`, without explicit rejection. |
| `prefetch.py:1072–1093`: hint indices0..min(D,T)−1 for A/B/SFA/SFB | Cache-prefetch gap | Prologue bounds protect short K; four operand hints per K tile. |
| `prefetch.py:1141–1160`: after load k, hint k+D while k<T−D | Same gap: rolling hint schedule | T5,D4 hints0/1/2/3 initially, then4 after load0. No out-of-range hint. |
| `prefetch.py:1095–1140,1162–1170`: acquire AB slot, actual TMA copies, advance state | `async_transfer@1` controller,after_slot_released,barrier_wait; pipeline rotation is separate | Copies retain readiness/slot-release obligations. Hint calls do not satisfy the producer barrier. Exact barrier consumers and bytes remain next. |
| `prefetch.py:1033–1036,1096`: pipeline state survives persistent output-tile iteration; reset_count each tile | Existing rotation with persistent state glue | `python/CuTeDSL/cutlass/pipeline/helpers.py:873–924`: reset only count; index/phase continue. Replay distinguishes resetting the full ring each output tile. |
| `prefetch.py:1840–1901`: stage count derives from SMEM capacity, occupancy, C storage and AB/SF bytes | Existing resource/stage choice, not independently measured optimum | S=floor((capacity/occupancy−1024−Cbytes)/ABbytes); epilogue stages then consume residual space. Exact layout sizes remain next. |
| `prefetch.py:761–765` versus baseline `:750–754` | Descriptor prefetch is a different event | Both versions already prefetch TMA descriptors. Data hint toggle must not be confused with descriptor setup. |

`python/CuTeDSL/cutlass/cute/algorithm.py:658–709` accepts a TMA CopyAtom and
partitioned source, emits a CuTe prefetch operation and describes global-to-L2
prefetch. The shared barrier pointer passed while unpacking the atom is a dummy,
not a completion protocol for actual shared copies. The compiler's final PTX lowering
and target behavior remain untraced. Do not infer a measured latency benefit.

The source-equation witness T5,S2,D4 is deliberately a scheduling-only binding,
not yet a legal whole-kernel GEMM configuration. Actual AB destinations are slots
0/1/0/1/0; hints reach four tiles ahead while consuming no additional shared slot.
This separates cache residency hints from P in the shared pipeline. Removing hints
preserves actual load indices; cache availability and execution time are separate.

[Retained checks](checks/README.md) cover T0..64,S1..8,D=auto,−2,−1,0..16 across
three persistent output tiles. They establish hint bounds/coverage and state indices,
not hardware completion. Next: finish dtype/layout/shape gates, derive one legal
GEMM binding, trace SFA/SFB to TMEM/MMA and output tail, then decide whether the
cache-hint gap warrants a new Template and paired ProofOutline. All other library
families remain open in [inventory](inventory.md).


## Second slice

The [contract/scale/epilogue trace](trace-blockscaled-layout.md) now supplies a
frontend-valid M144/N208/K320/L2 instance, both scale-window geometries and N256
TMEM overlap. Cache lookahead is proposed separately as `cache_prefetch@1` with a
paired outline. The original candidate-gap entries above document why that identity
was needed; they are resolved by the delta below, not by overloading pipeline P.
