# NVIDIA library inventory

Inputs are fixed in [the source manifest](inputs/sources.json). Paths below are relative to
the pinned Humming revision. This inventory precedes extraction. Status: full source file
set reconciled; all groups traced and closed with explicit remaining gaps. A listed branch is not counted as an
explained branch until its trace and obligations are recorded.

## Public and build boundaries

`humming/ops/__init__.py:1–33` exports GEMM, input processing, six weight operations,
the compute benchmark and four registration/launch helpers. `forward.py`, `layer.py`,
`transform.py` and the schema adapters compose these operations; they add preparation and
caller contracts rather than additional device entry definitions.

The build is JIT CUDA C++, not a TileLang/Triton kernel family. `humming/build.py:65–164`
builds the native launcher, device probe, CUBIN patcher and NVRTC driver. `pyproject.toml`
ships CUDA headers as package data. `humming/jit/runtime.py:76–111` and
`humming/jit/compiler.py:225–285,310–352` bind generated code and `kernel_expr` to a compiler.
`humming/csrc/launcher/launcher.cpp:415–452` registers GEMM and input processing with
PyTorch; auxiliary builders launch their loaded functions via the CUDA driver themselves.
`ops/utils.py:120–147` also registers Python custom operations, including FakeTensor
metadata paths. Those metadata paths establish shapes, not device execution.

## Mechanism groups and traced witnesses

| Group | Entry / body and registration | Representative binding | Shared variants and differences requiring disposition |
|---|---|---|---|
| G01 warp MMA GEMM | `ops/gemm.py:45–100` → `kernel/humming.py:128–202` → `include/humming/kernel/humming.cuh:32–237`; MMA body `mma/wmma.cuh` | SM80, FP16 A / unsigned INT4 B, N=128, K=128, block=(16,128,64), warp=(16,32,32), explicit MMA, no Stream-K | SM75 instruction shape; SM8x/89 gates; synchronous and warp-specialized ownership; native mixed formats on SM12x; FP16 vs FP32 accumulation |
| G02 warpgroup MMA GEMM | same builder; `kernel/humming_ws.cuh:31–260` or synchronous body; `mma/wgmma.cuh` | SM90, FP16 A / INT4 B, explicit WGMMA, block=(64,128,64), warp=(64,32,64), 4 stages, producer/consumer roles | SS vs RS operands, packed-K, late scales, split issue/wait, ring continuation, TMA/cp.async, H 20 policy |
| G03 tensor-memory GEMM | builder selects `kernel/humming_umma.cuh:31–440`; `mma/umma.cuh` | SM100, FP16 A / INT4 B, block=(64,128,64), warp=(64,32,64), 3 stages, CTA group=1, one dequant warpgroup | SS/raw weights vs dequantized operands; 1/2 dequant warpgroups; CTA group=2; TMEM accumulator lifetime; separate scale/weight barriers; indexed epilogue overlap |
| G04 block-scaled warp MMA | common/specialized body, `mma/mxmma.cuh`; `config/mma.py:120–155` | SM120, FP4 A/B, group=32, E8M0 scales, FP32 accumulator, explicit MXMMA | FP8/FP4, scale vector 1/2/4, mixed native operands, compiler gates, zero point disables Stream-K |
| G05 tile and expert scheduling | `include/humming/scheduler.cuh:1–423` consumed by all GEMM bodies | dense output grid, then indexed top-k and grouped expert descriptors | Dense raster order; Stream-K slices; indexed gathers; contiguous/masked expert groups; multicast and cooperative CTA ownership |
| G06 epilogue and partial combination | `include/humming/epilogue/{pipeline,smem_reducer,smem_writer,gmem_writer}.cuh`; `arith/epilogue_arith.cuh` | split K across two warp K partitions, then Stream-K slices with lock-ordered combine | Scale/bias placement, padding, indexed scatter, TMA stores, shared-memory reuse, final tile drain, TMEM output |
| G07 load/convert/scale pipeline | `memory/g2s_pipeline.cuh`, `memory/s2r_pipeline.cuh`, loaders, `datatype/dequant*.cuh`, `arith/{mainloop_arith,exp_offset}.cuh` | FP16 A / INT4 B, two unequal K-group scales; enumerate all 16 codes | Integer/float bit widths 1..8, native conversion, patched ldmatrix, zero point preprocessing, exponent rebasing and fused E8M0 scale |
| G08 inference matrix repack | `ops/weight.py:132–206` → `kernel/repack_weight.py:19–69` → `kernel/process.cuh:130–323` | INT4 B, A width 16, N=64/K=64, unpacked codes, mode=3 | Packed input extraction; 3/5/6/7-bit compression; sign-bit interleave; preprocessed INT→FP and ZP; mini-block transpose; packed-K; native dequant; expert axis |
| G09 contiguous bitstream pack/unpack | `ops/weight.py:49–64,209–224` → `kernel/{pack,unpack}_weight.py` → `kernel/pack_weight.cuh:53–103` | 32 unsigned 3-bit codes into three int32 words and inverse | Widths 1..8, cross-word carries, full code groups and inactive final CTA threads. Independent from G08 matrix permutation |
| G10 weight quantization / format decode | `ops/weight.py:22–47,67–129` → `kernel/{quant,dequant}_weight.py` → `kernel/quant_weight.cuh:178–307`, `kernel/dequant_weight.cuh:6–52` | FP16 group=128 to unsigned INT4 with integer ZP; decode sign-magnitude e0m3 separately | Symmetric/asymmetric; integer/floating ZP; negative scale; UE8M0 scale rounding; integer/mini-float/e0mX; FP16/BF16/FP32 inputs |
| G11 input transformation and scale phases | `ops/input.py:13–115,118+` → `kernel/process_input.py:83–207` → `kernel/process_input.cuh:48–301` and its helper headers | dynamic per-token INT8, then tiled dynamic group-token scales | Raw copy, activation, Hadamard; quantization modes; collect/quantize two-stage; group-token finalizer; row/scale layouts; expert/scatter/valid-token layouts; PDL ordering |
| G12 MXFP4 W4A8 preparation | `ops/weight.py:227–254` → `kernel/process_mxfp4.py:17+` → `kernel/process_mxfp4.cuh:40–69` | one 32-code FP4 group with delta exponent offset=1 | delta=0 bypass; negative-zero canonicalization; nearest rounding ties choose upper candidate; group tail guard |
| G13 isolated tensor-core benchmark | `ops/bench.py:8–45` → `kernel/tops_bench.py:26+` → `kernel/tops_bench.cuh:20–50` | SM80 FP16 MMA, repeat=65536, unroll=64 | WGMMA fence/commit/wait; MXMMA scaled instruction; accumulation type. Reuse instruction-selection rule; reported TOPS requires separate actual timing |

These are source-analysis bindings. `trace-families.md` substitutes selected dtype, tile,
instruction, scale or resource bindings per group. Its open boundaries are not records of
compiled or measured kernels.

## Dispatch and architecture coverage

`tune/__init__.py:26–58` registers SM75, 80, 86, 87, 89, 90, 100, 103, 110, 120 and 121,
with major-version fallback. H 20 has a distinct SM90 policy; PPU SM80 is a separate backend
and outside this assignment's NVIDIA path. `config/mma.py:120–127` defaults SM90 non-INT4
to WGMMA, SM12x block-scaled layers to MXMMA, supported layers to UMMA, otherwise MMA.
These defaults are distinct from explicit overrides and subsequent validity guards.

`kernel/humming.py:49–55` selects one of three mainloop headers: UMMA takes precedence,
then warp specialization, then the common body. `kernel/humming.py:258–302` distinguishes
native ldmatrix conversion and CUBIN patching; undocumented FP encodings are gated by
`config/mma.py:43–79`. The source's support claim and a compiled menu do not establish
that a format ran on any particular target in this study.

## Reconciliation and next action

Run `python3 knowledge/studies/humming/checks/inventory.py` from the KTA root. It checks
every pinned file's digest and file-set equality, parses every public export and JIT builder,
finds every `__global__` declaration in every device header, and assigns every header an owner.
The [machine inventory](checks/inventory.json) retains all compile-time branch/assertion
anchors for a source cross-check. Shared utility headers belong to their consuming group,
not to extra operators. Branch-anchor presence earns no semantic coverage credit.

Closure: `trace-families.md` contains D01–D53; `checks/coverage-result.json` reconciles every
source owner and decision. `checks/variant-dispositions.json` points to the external branch
ledger: every anchor remains partial at the compiled-branch boundary, linked to its group
and next evidence. A representative is not treated as proof of all its specializations.
Independent review is a separate next stage; no entry was added to the reviewed index.
