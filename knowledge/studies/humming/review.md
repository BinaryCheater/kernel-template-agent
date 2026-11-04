# Humming Template extraction review

**What you can use.** This study contributed four candidate rules. Choose one by the operation in your kernel:

| When your kernel needs to… | Rule and decision |
|---|---|
| Store already-quantized codes compactly | [Pack a bitstream](delta/bitstream_packing@1/template.yaml). Bind the code width and derive the packed word count. |
| Convert a group of values to integer codes | [Quantize a group](delta/groupwise_affine_quantization@1/template.yaml). Bind the requested format and produce codes plus scale and zero point. |
| Apply a normalized Hadamard transform | [Distribute butterfly stages](delta/butterfly_linear_transform@1/template.yaml). Choose values per lane and participating threads. |
| Combine several blocks’ contributions to one tile | [Initialize then accumulate](delta/first_writer_atomic_combine@1/template.yaml). Bind contributor count and target-supported update method. |

For example, a consumer requiring 3-bit codes can compose group quantization with bitstream packing. A group of 32 codes occupies three 32-bit words. See the [two-step chain](delta/quantizer-chain.yaml) for an actual source binding. Each rule carries its own usage instructions. The adjoining proof outline and checks explain the evidence.



A worked example: ordinary A16/B4 GEMM composes weight prepacking, a shared-memory copy,
a half-group register load and nibble conversion into the MMA B fragment. Every one of the
8192 coordinates in a 128x64 stage matches the expected B coordinate. Logical weight(17,43)
reaches lane 5 of N-warp 0/K-warp 1; its packed word is533 at N128 stride. All 16 codes decode
exactly at three zero points. Two unequal group scales reject the incorrect transformation
that moves arbitrary group scales after the whole K reduction.

The remaining decisions are explicit:

- UMMA needs a fuller role/TMEM ownership contract; MXMMA needs scale-operand fields.
- Odd-bit, native and exponent-compensated representation paths need more coordinate and
  numeric checks before widening the old prepack/decode rules.
- Asynchronous barriers, atomics, PDL, output reconstruction and rounding need exact compiled
  GPU evidence. The current checks recover source choices and arithmetic boundaries.
- The primary GEMM chain is partial and unpriced. Group/token factorization and MXFP4
  re-quantization retain their numerical obligations.
- The optional static WP check found existing WP4/WP5 lock mismatches. Per the clarified
  source-study scope, unrelated WP gates, delivery paperwork and commits are separate work.
  The original acceptance lock is preserved.

## Proposed knowledge

Four new rules have versioned ProofOutlines: **bitstream packing**, **groupwise affine
quantization**, **normalized butterfly transformation**, and **first-writer atomic combine**.
They distinguish choices that the nearest existing rules cannot express. Five CUDA source
Realisations and two explicit composition chains accompany them. Their invalid-choice
witnesses and current-schema checks pass; assurance remains declared.

## Coverage and evidence

The full pinned NVIDIA source family is inventoried: 13 public exports,10 JIT builders,
12 device entry declarations and 63 CUDA headers. The study traces thirteen mechanism groups,
including all three GEMM mainloop bodies, block-scaled instructions, expert scheduling,
epilogues, input phases, weight preparation and the compute benchmark.

The 53 design decisions close as20 explained,31 partial and 2 open. Every group has concrete
bindings, source locations, an outcome and next evidence. The 1137 compile-time branch/assertion
anchors are retained separately as partial; their presence earns no proof credit. Fifteen
schema-valid Gap records identify the missing evidence or expressiveness. This source-study
coverage is input to an independent review, not a claim that every specialization is correct.

Selected checks include1152 one-hot code bits across widths 1–8,4096 prepack coordinates,
8192 composed B coordinates,873 atomic completion orders across two generations, full
Hadamard basis vectors through H 512, FP4 rebasing and quantizer boundaries, and dense SM90
policy/fitting at ten M values. SM75/80/90/100/120 source instruction bindings are recovered.
All study replays, schema/reference/domain checks and Ruff pass. No GPU measurement ran.

Reading paths:

- [Inventory](inventory.md), [instance trace](trace.md) and [all family decisions](trace-families.md).
- [Pool delta and composed bindings](delta/index.md), including the specific next evidence for each gap.
- [Reproducible checks and retained results](checks/README.md), [correction pass](checks/correction-pass.md), and [complete gate output](checks/static-gate.txt).

The requested source-study artifacts are delivered at proposed knowledge. The gaps remain
recorded for independent review and subsequent evidence. No reviewed-index or live Store
admission, GPU validation or WP implementation completion is claimed.
