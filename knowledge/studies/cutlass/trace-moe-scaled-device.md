# Scaled MoE: accumulator overlap and final scaling

When accumulator copies and scale fragments compete for fixed tensor-memory
columns, inspect the **intersecting regions**, the read direction and the exact
release event. A one-stage readiness pipeline may alternate two physical regions;
a pipeline-stage number alone does not identify the region. This source instance
reuses the earlier region-lifetime inequality and retains the traversal choice as
explicit family glue.

This trace completes `torch_scaled_grouped_mm.py` in the five-file
[MoE inventory](inputs/moe-variants.json), together with the
[dense trace](trace-moe-dense.md) and
[scale preparation trace](trace-moe-scaled-metadata.md). Paths and lines refer to
the retained tree in [the source manifest](inputs/sources.json). Completion here
means source extraction with executable scalar checks and named implementation
gaps, not a working native kernel or admission.

## Selected branch and frontend

The MXFP8 witness uses E4M3 A/B, E8M0 metadata, BF16 C, FP32 accumulation,
sf_vec_size32, MMA128×256×128, cluster1×1, no accumulation into old C,
separate initialization and independently padded scales. Use four token lengths
[32,0,160,64], hidden256/intermediate256, with real and padded prefixes from the
[metadata binding](inputs/moe-scaled-metadata-binding.json). Forward reduction K256
is fixed; backward reduction lengths vary by expert, retaining empty expert1's
output. The N256 branch activates overlapping accumulators.

The selected helper `utils/blackwell_helpers.py:1418–1466` binds MXFP8 to
MmaMXF8F6F4Op with native K32; MXFP4 and NVFP4 select their respective FP4 atoms
with native K64. The source constructor accepts per-CTA M128, N64/128/256, K
multiples of4×sf and sf16/32. Two-CTA instructions require MMA M256 and even
clusterM; cluster dimensions are powers of2, each at most4, product at most16.
This shape menu does not establish descriptor alignment, native operand/scale
formats, packed-FP4 DLPack interpretation, legal stage counts or TMEM capacity.
CLI format configuration pairs sf32/E8M0 with MX formats and sf16/E4M3 with NVFP4.
Direct callers must preserve those pairings. Bias and forward output accumulation
are rejected. Scaled fixed_expert_cnt is stored but not consumed; the host computes an
is_dynamic_expert_cnt flag but marks every tensor layout dynamic through its
helper. No static-specialization effect or alternate scheduler is established.

## Device decisions and rule boundaries

| Decision and source | Source binding | Rule field or gap |
|---|---|---|
| Setup attributes:280–441; stage policy:448–522 | SFB atom is always one-CTA and rounds N up to128; TMEM SF columns derive from instruction-K repetitions. AB+SF stages share a SMEM budget, reserving output/barrier/scheduler storage; output stages use the remainder. | Derived resource policy, not free stage knobs. Existing multi_buffer_rotation@2 ends at8. Physical layouts, native capacity, positive stages and any larger derived stage count need their own supported binding. |
| Pipeline:1393–1456; TMA:1622–1788 | Four TMA copies A/B/SFA/SFB use the same stage barrier; tx_count is the sum of all four byte counts times atom CTA count. AB tile-local count resets but ring generation persists. | async_transfer@1's global→shared controller completion applies conditionally to each operand. No independent thread-copy completion channel is selected here, so byte_arrival_completion_join@1 is not the explanation. |
| MMA:1818–1849,1901–1948 | After AB readiness, copy SFA/SFB SMEM→TMEM, then call GEMM with paired [data,scale] operands. First K tile sets ACCUMULATE=false. | Native block-scaled MMA is a selected backend binding with open lane/format/tail obligations. DeepGEMM block_scaled_partial_accumulation@1 resets raw unscaled partials and promotes them later; it does not describe scales consumed inside each MMA here. SMEM→TMEM copy also is not an async_transfer@1 global load. |
| SFB N64:1738–1742,1888–1899 | Tile pair shares a128-wide SFB window: slice_n=tile_n//2; selected TMEM pointer shifts2 columns for odd tiles. | Same logical N64 parity/window instance seen in earlier block-scaled traces. Scalar logical window origin plus selected half recovers tile_n×64. Exact two-column TMEM/lane interpretation and two-CTA slice lowering remain implementation obligations. |
| Compressed layout:396–423,1487–1504 | N256 uses two physical accumulator regions of width256 with stride256-SF_columns, but only one pipeline stage. Producer stage=phase^1, consumer stage=phase. | Existing physical_storage@1 pre_lifetime applies per intersecting region. It does not choose compressed strides, alternating region identity, traversal or source release threshold. No complete capacity model is inferred from a rule name. |
| Epilogue traversal/release:2078–2138 | Region0 reads subtiles backwards; region1 reads forwards. Release at loop index SF_columns//epi_N after fencing TMEM loads, then continue reading only the nonintersecting remainder. | Source mapping and region-lifetime instance, not a new Template. For MXFP8 K128, SF_columns48, regions[0,256) and[208,464), SF[464,512); epi_N32 releases after two subtiles. Source threshold is floor(SF/epi_N), not universally ceil(SF/epi_N)-1; exact-divisibility cases deliberately read an extra subtile before release. Native fence/completion/overlap progress remains open. |
| Empty K:1904–1950,2090–2177 | K0 does not acquire/commit/advance accumulator readiness; backward starts register output at zero and skips TMEM reads. C output/subtile clock still advances for every work tile. | Same separate-clock and identity instance as dense MoE. Producer/consumer phases remain paired across consecutive K0 work. For NVFP4 the later alpha multiplication requires a finite computed alpha to preserve zero; source does not validate it. |
| Global scaling:2097–2108,2141–2150 | When global_scale_a exists, load both experts' Float32 globals, multiply them into alpha, multiply accumulator by alpha, then cast to C. If A global is absent, B global is ignored. | operator_fusion@1 consumer_kind=elementwise/intermediate_site=registers/fused_ops=scale_and_cast supplies the tile-local epilogue. It does not authorize factoring input scales through a reduction. Require paired arrays and an admitted numeric policy; finite input factors alone do not prevent their FP32 product from overflowing. |
| Output:2110–2198 | Traversal-selected real_subtile_idx controls both TMEM reads and destination coordinates. C ring advances in loop order and drains before shared storage/TMEM is freed. Optional backward reduction store casts before updating C. | staged_async_output_store@1 source-ring lifetime is reused on the overwrite path with native descriptor/source-read/write-completion premises. Existing-C reduction arithmetic remains the separate family variant/gap from dense MoE; first_writer_atomic_combine@1 cannot supply it. |
| Scheduler:1522–1575 | Same initial publication, valid loop and terminal publication as dense MoE; final commit does not advance producer state before producer_tail. | The dense sentinel/tail protocol finding applies unchanged, including initially idle workers publishing two sentinels. Index coverage is not native progress evidence. |
| Host inputs/reference:2789–2869,2949–3134,3196–3377,3553–3596 | FP4 correctness data uses{0,1,-1}, scales and NV globals use{1,2}; performance input bytes differ. Manual reference scales FP32 operands before matmul. Validation uses exact comparison, patches backward empty reference output, and skips the forward reference check when total_tokens//experts=0. | Test-data fixtures and reference workarounds, not production quantizers or broad numeric evidence. No reference success is inferred for the skipped branch. Native packed shape interpretation, reference precision and finite/global/tail contracts remain explicit. |

## Geometry and numeric counterexamples

The source shape validator accepts MXFP8 MMA128×256×768, sf32, cluster1×1. Native
K32 implies24 instruction repetitions and SF_columns=(4+8)×24=288. The compressed
accumulator stride becomes-32 and the early-release index9 lies outside the eight
N32 subtiles. This violates region/allocation and release assumptions even though
the frontend accepts it. It is an executable source/resource gap, not evidence of
a GPU failure. No old rule scope is widened to admit that geometry.

Scale relocation also needs a numerical contract. Put raw A=(3,-2), B=(2,3), block
scales1, global A=1+2^-23, global B=1+3×2^-23. All data values are valid NVFP4.
Raw dot is0, so final multiplication produces0. If each input is first globally
scaled into FP32, a scalar FP32 multiply/add dot produces2^-21; even the exact dot
of those stored FP32 operands is nonzero. The difference survives BF16 output.
This distinguishes numerical order from exact-real distributivity. The scalar
arithmetic model is not an assertion about a specific native matmul's TF32/FMA
policy. Source fixtures with integer global scales do not settle this contract.
For K0, the retained zero-then-alpha branch also gives NaN when alpha is infinite;
therefore the finite-alpha premise is consequential.

## Composition and evidence

The [metadata chain](delta/moe-scaled-metadata-chain.yaml) establishes scale regions
and immutable transfer descriptors. The scalar expert scheduler selects matching
real-data and padded-scale origins. External native staging/MMA then produces a
defined accumulator for positive K; backward K0 instead supplies register zeros.
Region-aware traversal retires overlap before the next destructive write, while
remaining values survive in their nonintersecting region. Tile-local final scale
and cast precedes the output source ring. This composition retains external edges
for native layouts, publication, numerical policy and source scheduler correction.
There is no new candidate identity in this slice.

[Retained device-slice checks](checks/moe-scaled-device-result.json) execute the
actual shape validator across768 cases (105 accepted/663 rejected), the forward
accumulation guard, derived source region expressions and epilogue size helper.
120 legal region domains/read traces establish overlap retirement under the
stated order model.243 five-work K0/positive-K streams preserve paired physical
phase selection and distinct output clocks. The retained zero/load/scale branch,
N64 half selection, bad K768 geometry and FP32/BF16 counterexamples are checked.
Native SSA, fence/descriptor semantics, operand lane maps, kernel progress,
accumulation precision and performance remain open. No GPU execution, Realisation,
ProofSummary or Store admission occurred.
