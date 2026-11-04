# Grouped FP8 promotion, metadata windows and fused output

Source: retained [tree](inputs/sources.json). The native grouped representative
is `flashinfer/gemm/kernels/grouped_gemm_contiguous_blackwell.py`; the fused
gate/quantized-output route lives in `flashinfer/gemm/cake_grouped_fp8_fused_silu_quant.py`
and its JIT-selected CUDA under `csrc/cake_grouped_fp8_fused_silu_quant/`.

## Four scale steps share a transfer, not a mathematical scale

Retrieval cue: per-K metadata is contiguous but staging one scalar per step
produces many scattered loads/publications. New
[multi_step_metadata_staging@1](delta/multi_step_metadata_staging@1/template.yaml)
and [proof](delta/multi_step_metadata_staging@1/proof.yaml) make that grouping
independent of block_scaled_partial_accumulation@1's raw-dot extent.

| Source choice | Candidate field / rule | Binding |
|---|---|---|
| Grouped kernel :203-210,3301 | `steps_per_window` | Source chooses G4 when the number of128-deep K blocks is divisible by4, otherwise G1. It does not drop an incomplete group or silently promote a256/512-deep dot with one scale. |
| :931-955 | `slice_identity`, storage | Shared SFA offset=slot*(128*G)+row*G+j; SFB offset=slot*G+j. FP32 source K steps are contiguous, so G4 uses one16-byte cp.async per row. Actual alignment/stride is a premise, not established by an `.align()` annotation alone. |
| :1334-1392 | `whole_window` publication | Acquire one metadata slot, copy every G-step row/slice and SFB, commit once, then advance the metadata producer. Payload identity includes expert, output tile and K group. |
| :2050-2110 | `separate_arithmetic`, `window_lifetime` | Wait once for metadata, then for each j independently wait/read its raw MMA partial, apply SFA[j]*SFB[j], accumulate FP32 and release that partial. Release the metadata only after all j/readers finish. |
| :426-441,2848-2951 | `stages` / resource coupling | Metadata ring uses10 slots atG1 or3 slots atG4. Arithmetic/TMEM accumulator and output-store rings have separate counts. G4 triples4-step windows rather than treating three scale slots as three arithmetic steps. |

For128 SFA rows and one SFB row, G4/S3 metadata uses4*3*4*129=6192 shared
bytes before alignment/pipeline overhead. This is the new candidate's storage
formula, not the whole kernel's shared footprint. More metadata bytes per slot
trade against fewer publication/copy groups; speed is an implementation question.

vectorized_access@1 supplies the4/16-byte aligned transfer mapping, and
per_slot_handshake@1 supplies the full/empty idea. The latter's source-domain
stage cap is8, so this source's10-slot G1 ring is not an exact old-rule binding.
Neither older rule describes several independent arithmetic-step slices within
one metadata-slot lifetime. The new candidate retains those slices explicitly.

## Promotion and final-store ownership remain separate choices

The same kernel :187-242 distinguishes three organizations:

| Work / organization | Existing binding | Source boundary |
|---|---|---|
| Ordinary multi-K promotion and separate epilogue | block_scaled_partial_accumulation@1, physical_storage@1 | Read each raw128 partial from TMEM, promote with its own scales into FP32 registers, then transfer the completed promoted tile through final TMEM for epilogue readers. |
| One128-deep K block | operator_fusion@2, consumer_kind=elementwise, intermediate_site=registers, fused_ops=scale_and_cast, checkpoint=none | No repeated promotion sum exists; scale/cast/store may consume the single raw accumulator directly. Retaining an unrequested intermediate is unnecessary; externally required consumers would change the checkpoint premise. |
| Multi-K `fused_deepk` | Same final scale/cast fusion binding | Accumulator-update threads retain the full promoted FP32 result in registers through the final store, removing the register-to-final-TMEM-to-epilogue handoff. :2210 onward stages output through shared memory and TMA, with complete store-buffer acquire/fence/retirement. |

:221-230 allows one or two accumulator warpgroups. :1975-1982 derives each
group's contiguous N-subtile ownership. This changes register ownership and
latency-hiding opportunities, not the decoded contraction. Sharing the output
store responsibility still requires every uniquely owned subtile exactly once.
`num_acc_stage=4` at :444-449 reuses TMEM space freed by removing the promoted
final buffer; do not infer independence between fused ownership and accumulator
stage capacity. Performance comments are hypotheses/source context here, not
measurements from this study.

## Fused gate/up and output quantization preserve checkpoints

The public prepared wrapper documents the unfused reference chain:

1. GEMM produces BF16 gate/up halves.
2. SwiGLU consumes those BF16 values and rounds its result to BF16.
3. Each row's128-column group computes max(abs(h)), floors it at1e-10,
   writes FP32 scale=max/448 and E4M3 codes of clamped h/scale.

Native representative
`cake_grouped_fp8_fused_silu_quant_a54057945ffaff1016b1_kernel.cu:869-881`
rounds gate/up to BF16 before fast sigmoid/multiply, then rounds h to BF16.
At :948-978 each accumulator warpgroup owns64 h values; the two groups exchange
maxima through shared memory and a256-thread barrier before deriving the common
128-column scale. A local64-value quantizer is not the same output format.
The generated code removes global intermediates while preserving their logical
conversion stages. Fast exp/reciprocal and ordered GEMM accumulation remain the
numerical binding needed for any bitwise-chain claim.

operator_fusion@2's `consumer_kind=row_reduce_then_map`, registers/shared
intermediate and full-dependency-local premise describe the group quantization
edge once the128-column dependency is joined. Its scale_and_cast choice does
not by itself define the preceding gated map or quantizer recipe; those are the
explicit operations above. The complete native SwiGLU/E4M3 chain is retained
as this source binding rather than being hidden behind a generic fusion label.

Routing supplies further organizations: large complete expert block pairs,
separate odd-tail units, a mixed two-CTA schedule, or grouped GEMM plus a fused
activation/quantizer launch. Metadata-known route models are calibrated only
to their declared geometry/device. Their fitted constants and reported speedups
are not extracted as universal scheduling laws. Full expert/input/scales and
odd-tail bounds remain the chosen route's ownership premises.

## Evidence and next work

The [logical check](checks/check_metadata_windows_and_fused_casts.json) uses
identifiable(row,step) metadata, unequal per-step scales, a premature-overwrite
counterexample, two feature owners, cross-output-tile ring wrap and G1 tail
fallback. BF16 witnesses distinguish gate/up and activation checkpoints; a
64+64 split verifies that quantization needs the full128 maximum. Native
cp.async/UMMA, exact subgroup stores and fast activation bit equality remain
implementation obligations.

The grouped FP8 promotion/epilogue design now has concrete dispositions. Continue
SVDQuant/low-rank and mixed BF16×FP4 paths, task-scheduled routed operands and
remaining quantized output recipes, then deterministic grouped training and
full-library reconciliation.
