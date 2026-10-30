# WP7 delivery: first real chain, FP16 GEMM (code stage)

Branch `wp/WP7`, cut from `integration` @ `90aa678` (WP0–WP5, WP10a, WP13); `integration` @ `65a0e2d` (D320, code-stage gate) merged. This is the **code-stage** delivery (D320: the user paused real data and measurements on 2026-10-31). Nothing below has run on spark yet. Every real-run item is listed under "Pending real runs", and WP7 is not accepted until `KTA_DATA=~/kta-data scripts/gate WP7` (full) passes. Evidence: `deliveries/WP7.evidence.json` (unit runs only, commit `290d6fd`).

## What was done

All of the following is code and data in the repository, checked by unit tests only (claim 0). None of it is a real run.

- **Contraction and operand traffic now have costs (D280, fixes WP5's known issue).**
  - WP10a's proposed `output_tiling` and `compute_instruction_selection` enter the seed as version 2 (`knowledge/seed/templates/`).
  - `output_tiling@2` costs launch slots (serial), the L2 bytes each tile reads, and compulsory DRAM bytes.
  - `compute_instruction_selection@2` costs tensor FLOPs and instruction issue per executor tile.
  - Γ of `compute_instruction_selection@2` compares the operand type, from the Problem's `operand_bits` and `operand_exp_bits`, with the instruction's (`p.instr_*`).
  - The `contract` and `map` Patterns declare the new numeric params.
  - On the test HwModel the shortlist is no longer tied: tile shape, wave quantization and launch slots separate candidates (`test_formulation_costs_contraction_and_traffic`).
- **Family `knowledge/seed/families/gemm_fp16_tilelang/` (D282).**
  - Seven chain items: the five seed GEMM steps plus the two Templates above. Every item has `kta:step` begin/end markers in the generated `kernel.py`, and `validate_family` reports no issue.
  - `gen.py` emits a TileLang `@T.prim_func gemm(A, B, C)` per assignment and shape: T.Kernel grid, shared A/B tiles, `T.Pipelined(num_stages=S)` with cp.async copies, `T.gemm` (mma.sync m16n8k16, FP32 accumulator), and a clipped store.
  - It returns a `GeneratedKernel` with grid, block and dynamic smem, which is the form WP6's `pipeline.generate` accepts.
  - Variables: BM, BN, BK, S, threads and split.
  - Glue constraints: `P == S - 1`, accumulator registers, and `split == 1` (no cross-CTA combine).
  - The space has about 4.2·10⁷ points, of which about 70 per shape are feasible, so the solve enumerates exhaustively in about 10 ms.
- **Case (`knowledge/seed/cases/gemm_fp16.yaml`, D289).**
  - Shapes: M=N=K=4096, 8192×1024×4096 and 1024×1024×16384.
  - Contract per docs/design/50 §3:
    - FP16 rtol = atol = 1e-2, rel_l2 1e-3;
    - input distributions normal, zeros, extreme, cancellation and `large_dynamic_range`;
    - edge shapes 1×1×1, 129×193×65 and 1×4096×4096.
  - Baseline: `torch.matmul(A, B, out=C)`.
  - `gemm_bf16.yaml` is the same Case with BF16 operands, used for `reuse_plan`.
- **Measured peaks and the HwModel (D283).**
  - `kta.measure.peaks` / `peak_kernels`: six CUDA microbenchmarks timed under the Protocol. They cover DRAM, L2, smem wavefronts, mma.sync FLOPs, FP32 FMA (which also gives issue in warp instructions), and CTA launch.
  - Output: one measured HwFact per unit, then an HwModel whose limits are the Target's facts plus `smem_queue_depth`, `smem_banks` and `smem_bank_bytes`.
- **torch.matmul baseline under the same Protocol (D284).**
  - `kta.measure.baseline` plus `scripts/torch_bench.py`, run in the PyTorch worker image on the leased GPU. It repeats the bench job's loop: input rotation, NaN fill, 2×L2 flush, event pair per call, the Protocol's minimums, and NVML samples.
  - The Measurement statistics come from the same code as for kernels (`Measurer.measurement_body` / `write_measurement`, split out of `_record` without changing values).
  - The baseline itself is a Report fixed before measuring and named by `Case.refs.baseline`.
- **TileLang backend additions (D281).**
  - A `PASS_CONFIGS` dict in `kernel.py` is applied through a `PassContext`.
  - The codegen reports `dynamic_smem` and `grid`. A LaunchSpec with less dynamic smem or another grid fails the Build.
  - `Build.smem` = static + dynamic smem, so occupancy feedback sees it.
  - The family disables warp specialization and TMA lowering, so copies stay cp.async; the harness cannot build host tensor maps.
- **Checks.**
  - `full_check(..., sanitizer_shape=...)` runs the sanitizers on a small shape (D285; WP7 uses 200×136×72).
  - New input distribution `large_dynamic_range`.
- **reuse_plan with an assignment (D286).**
  - `p.*` expression bindings are evaluated per shape for one assignment before Γ.
  - `problem_params` accepts a Problem record ID.
  - On BF16, exactly the `mma` step is invalid and the other six are kept; on FP16 all seven are kept (`test_reuse_plan_marks_only_the_instruction_step`).
- **Pre-WP6 driver `tools/kta_wp7` (D99).** Run as `PYTHONPATH=tools KTA_ENV=dev KTA_WP=WP7 uv run python -m kta_wp7 <step>`. Steps:
  - `setup`: measure-v2 (0.4% floor from WP3), the Case records, the Family/Chain records with `files/*` blobs (WP6's format), and the unit model;
  - `peaks`;
  - `topk`: per bucket, solve with the WP5 API, then build the shortlist in parallel, re-rank, run `full_check` on the top k, measure the passing candidates and the baseline, and compute Spearman's rho of predicted vs measured. It writes the Report `wp7_topk`;
  - `faults`: the three injections (D289), Report `wp7_faults`:
    - out-of-bounds: TileLang boundary legalization off, at 200×256×256;
    - racy legality: glue `split == 1` replaced by `split >= 1`. The solver admits split-K, the best admitted split-K candidate is built, and its CTAs store partial tiles to the same outputs;
    - cuBLAS: a generator emitting a host library that calls `cublasGemmEx`;
  - `reuse`: Report `wp7_reuse`;
  - `replay`: in a fresh process, recomputes from records only the solves, re-ranked predictions, Measurement statistics (from timing blobs and recorded seeds) and rank correlations. It writes a `replay` Report with `consistent`.
- **Gate (D288).** `wp7` in `tools/kta_gate/specific_wp7.py`, registered in `specific.py`, unit-tested in `tests/unit/test_gate_wp7.py`.
  - Static checks: the named unit tests; the family validates and uses all seven Templates; the Case file has the required shapes, contract and baseline.
  - Record checks: listed, real, clean judgement or Solution; three caught faults with failed GPU Checks; consistent replay; HwModel; baseline Measurement; reuse Report. These are WARN in the code stage.
- **Tests.**
  - `tests/unit/test_wp7_family.py` (7) and `tests/unit/test_wp7_measure.py` (7);
  - `tests/unit/test_gate_wp7.py` (6);
  - WP4 unit tests updated for seven seed Templates.

## Fixtures and simplifications

- No `# kta:fixture` added. Inherited, unchanged: `src/kta/store/bench.py` (`_body`, D156).
- The unit tests use an illustrative HwModel (`tests/unit/test_wp7_family.py`, `HW`): its peaks are not GB10 measurements, and its limits are WP2's probe values. Real peaks come from `kta_wp7 peaks` (pending).
- Allowed by WP7: none. The simple first case (D94) is the FP16 GEMM itself.

## Known issues

- **Nothing has run on hardware (D320).** The TileLang prim_func path for this GEMM on sm_121 is untested. That covers T.gemm under `tilelang.lower`, the pass-config keys, and the `dyn_shared_memory_buf` attribute. The first spark build may need generator fixes.
  - Supporting evidence, read only: a research-repo run of a similar TileLang GEMM on GB10 (`/Users/dingyi/Projects/agentic-codesign/kernel-lab/results/gb10-baselines-20261019-v1/gemm/tilelang/gemm/square/`) lowered to cp.async + mma.sync. With a `C_shared` epilogue it used a TMA store, which this family avoids.
- **Bottleneck-model ties remain** among assignments that differ only in variables touching non-bottleneck units (S, threads). Compile feedback (occupancy) separates some at rerank.
- **Mixed counting in `issue`.** `async_transfer@1` counts thread-level copy operations; the mma step counts warp instructions. The peak is in warp instructions. This is non-binding at 4096³ but should be settled in the method discussion.
- **`reduction_partition@1`'s combine FMAs** are charged for k-tile accumulation that happens inside mma. This overcounts, but the term is far from the bottleneck.
- **DRAM traffic is only the compulsory bytes;** operand re-reads are charged to L2. There is no block-rasterization step, so L2 misses are not modeled.
- **Wave-quantization slots** use shared memory and threads, not registers, which are unknown before the build.
- **Fault 1 is broader than a write.** Switching off boundary legalization makes residue tiles read and write out of bounds, so memcheck may report reads as well.
- **Fault 2 is a global-memory race.** It is caught by `numeric` (and possibly `anticheat.determinism`), not by racecheck, which covers shared memory only. TileLang inserts its own shared-memory barriers, so a shared race is not expressible through this family's generator.
- **The build cache key** (WP2's `build_key`) does not include the codegen script's hash. TileLang Builds cached before this change would be reused, though none of this family's kernels exist yet.
- **Old dev stores conflict on re-import.** Stores created before this branch (WP4/WP5 dev stores) hold the old `contract`/`map` Pattern bodies, so re-importing the seed there reports a conflict.
- **Baseline timing** includes the Python call overhead between the two CUDA events (a few µs). The bench job's ctypes launch has the same kind of overhead.
- **Replay determinism.** Replay re-solves with a fixed rng. That is identical only for exhaustive solves, which is the case for this family. It also needs the `job_id` that Measurements now record in their `params` blob (added here).
- **Spark settings recorded, not changed.** On spark the GPU performance counters are admin-only (D228) and clocks are unlocked. Changing either is system configuration.

## New decisions

D280 (needs user review), D281, D282, D283 (needs user review), D284 (needs user review), D285 (needs user review), D286, D287 (needs user review), D288 (needs user review), D289.

## Needed changes

- **WP6 (orchestrator):**
  - `work_checking` should pass `sanitizer_shape` (D285); otherwise racecheck runs on 4096³.
  - `_sass` passes Templates' `expected_sass` with the family assignment, but their `when` names Template variables such as `instruction`, which are bound to constants. Evaluate `when` over each item's bindings, as `kta_wp7.pipeline.sass_for` does.
  - `work_measuring` should measure baselines with `kta.measure.baseline.measure_baseline(m, baseline_id, case, shape, protocol_id, case_id)`. `_baseline_us` looks for `refs.case` + `refs.baseline`, which these Measurements carry.
  - `Case.target` is the Target name (D287).
  - `kta case add` can read `knowledge/seed/cases/*.yaml` (sections `problem`, `contract`, `case`, `baseline`; `kta_wp7.records.write_case`).
- **Additive changes to lower modules made in this WP** (owners: WP2, WP3, WP4):
  - `backends/tilelang` (gen.py, launch checks) and `common.write_build(dynamic_smem=)`;
  - `checks/full.py` (`sanitizer_shape`, `extra_dists` passed through `full_check`) and `checks/inputs.py`;
  - `knowledge/scope.py` (`reuse_plan(..., assignment)`, `problem_params` by ID);
  - `measure/runner.py` (split of `_record`, `job_id` in params), plus new `measure/peaks.py`, `peak_kernels.py`, `baseline.py` and `scripts/`.
- **Seed and tests of other WPs:**
  - seed Patterns `contract` and `map` gain params;
  - two seed Templates added;
  - `tests/unit/test_knowledge_pool.py` and `test_knowledge_use.py` (WP4, not locked) now expect seven seed Templates.
- **Spec edits:** WP7 implementation notes in `spec/modules/measure-check.md`, `target-exec.md` and `knowledge.md`; module READMEs of measure, checks, backends and knowledge.

## Pending real runs

Needed for the full gate (`KTA_DATA=~/kta-data scripts/gate WP7`), in order, with spark deployed at the commit (`scripts/worker-setup spark`, env `KTA_ENV=dev KTA_WP=WP7 KTA_DATA=~/kta-data`):

1. `kta init`, `kta worker register spark` and `kta target probe` in the WP7 store, then `kta_wp7 setup`.
2. `kta_wp7 peaks`: the measured HwModel.
3. `kta_wp7 topk`: builds, full checks and measurements of the top 8 per shape, the torch.matmul baseline per shape, and Spearman's rho, to be copied into this note.
4. `kta_wp7 faults`: all three caught; `kta_wp7 reuse`.
5. After WP6 merges: the full `kta solve` run to `accepted` or a recorded judgement; restart after a kill; `kta_wp7 replay` (and a replay of the solve loop) in a fresh process.
6. Evidence manifest with the record IDs; the full gate.

## Gate output

Run: `KTA_DATA=~/kta-data scripts/gate --stage code WP7` on the control host at the commit adding this note (code `290d6fd`).

```text
(pasted below after the run)
```
