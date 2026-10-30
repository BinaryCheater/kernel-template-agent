# WP3 delivery: measure and checks

Branch `wp/WP3`, cut from `integration` @ `c4081bb` (WP0, WP1, WP2, WP4, WP10a, WP13 merged). Code commits of the evidence runs (all clean, all on this branch):

- noise probe and `measure-v2`: `832061774429ff9502f33dd952202f537be70820` (claims 1, 2);
- compare and checker selftest: `818dff49cc047255a64c30411d870ef6770ff54a` (claims 3–6). It differs from `8320617` only in documentation. The first runs at `8320617` were recorded dirty because doc edits were uncommitted, so they were repeated and are not listed;
- the Protocol record `measure-v1`: written earlier at the clean commit `aa83ac2f64e5` (claim 0) and reused by identity.

Records are in `~/kta-data/dev/WP3/kta.db` on the control host (this Mac). Command outputs are under `~/kta-data/dev/WP3/evidence/<commit>/`. Claim numbers are 0-based indexes into `deliveries/WP3.evidence.json`.

## What was done

All runs: control plane on this Mac (`KTA_ENV=dev`), GPU jobs on spark (GB10, sm_121) through the Executor's ssh transport. Kernels are deliberately simple (D94, D233).

- **Measurement (`kta.measure`).**
  - The worker-side job `kta.measure.bench` loads one or two modules and times each launch with its own CUDA event pair (D220, D231). Before every iteration it:
    - rotates to the next of 4 input sets;
    - fills the outputs with NaN;
    - writes a 2×L2 flush buffer (48 MiB on GB10).
  - Warm-up runs until both minimums are met: 10 iterations and 50 ms. The measured phase does the same with 100 iterations and 200 ms.
  - NVML samples SM clock, max clock, memory clock, throttle reasons, temperature and power at the start and end of the measured phase. GB10 reports no memory clock, so that field is `null`. Clocks are not locked, because locking needs root.
  - Jobs are kind `measure`, so they run in the `gpu-exclusive` pool under the GPU lease.
  - The control plane computes the median and a bootstrap 95% CI (1000 resamples, seeded `ctx.rng` streams recorded in the `params` blob). It writes a Measurement whose `protocol` is the Protocol record ID, with refs `env`, `worker`, `build`, `protocol` and `problem`.
  - `compare` runs both modules in one job, in a random order per round. It writes two Measurements and a Report holding the bootstrap CI of the relative difference and the verdict: CIs disjoint and |diff| > 3 × noise floor.
- **Protocol records.**
  - `measure-v1` `6210a4eac4cc…` holds the spec's defaults plus a provisional 1.1% noise floor from WP13 finding 6 (claim 0).
  - `measure-v2` `86531d7af93f…` is v1 with the measured noise floor of 0.4% (claim 2, D226).
- **Noise probe on spark (claims 1, 2).** Each kernel was measured in 30 consecutive groups under `measure-v1`; each group is its own job (fresh process, context and allocation):

  | Kernel | Median of medians | CV of group medians | Deviation from the median: median / p90 / max | Report |
  | --- | --- | --- | --- | --- |
  | f32 vector add, n = 2^24 | 892.42 µs | 0.309% | 0.287% / 0.411% / 0.516% | `92ad89a64c80…` |
  | FP16 tiled GEMM, 1024³ | 884.47 µs | 0.346% | 0.003% / 0.704% / 0.911% | `1794b0b1b4b6…` |

  - **Noise floor: 0.4%.** It is the larger CV rounded up to 0.1%, and it is now `measure-v2`'s `noise_floor_rel`. The significance threshold is therefore 3 × 0.4% = 1.2%. WP13's capture-to-capture deviation was 1.1% median and 4.3% p90 (finding 6). Interleaving inside one job and repeating the protocol give a finer resolution.
  - Most of the spread is between jobs, not within one. Within a group the vector add's 95% CI is about 0.2–0.35% wide, but its group medians alternate between about 889 and 895 µs. The GEMM's medians step from 884.4 µs to about 890.6 µs in the later groups, while the GPU temperature rose from 43 to 51 °C. The SM clock sampled at group end ranged from 1748 to 2496 MHz (boost, not locked).
- **Interleaved comparison under `measure-v2` (claims 3, 4).**
  - FMA chains of 1000 vs 1050 steps differ by **+3.92%** (CI +3.92%..+3.92%): significant (Report `8c804cb2b53e…`). The pair was built to differ by about 5%. The measured 3.9% reflects launch and memory overhead outside the loop.
  - The same kernel against itself differs by 0.00%: not significant (Report `49287abc2f5b…`).
  - The degenerate CIs come from the event timer's 0.032 µs steps on very stable runs.
- **Checks (`kta.checks`).**
  - `check_numeric`: the module is run once per input distribution with NaN-filled outputs. The reference comes from `Problem.reference_impl`, a torch statement run in FP64 on the GPU inside the worker image (D223). The criterion is per element `|y−ref| ≤ atol + rtol·|ref|`, plus the relative L2 error.
  - `check_sanitizer`: compute-sanitizer memcheck and racecheck.
  - `check_anticheat`:
    - an nsys kernel trace (D221);
    - NaN coverage;
    - repeat-run determinism;
    - linked libraries and imported symbols, read from the module's ELF dynamic section (D230).
  - `check_sass`: opcode statistics and `SassExpect` checks; a violation writes a Gap with attribution `intent_sass` (D229).
  - Hidden shapes come from a private seed, and Check records show only their category (D225).
  - `full_check` combines the checks above with the Contract edge shapes, the hidden shapes, and a Contract-identity check against the Case (D227).
  - `verify_space` writes StatusChange(`tested`) for its subjects when every item passes.
- **Checker acceptance (claims 5, 6).** `full_check` ran on spark for a correct FP16 GEMM (128³) and three faulty variants:

  | Variant | Verdict | Failed checks | Aggregate Check |
  | --- | --- | --- | --- |
  | correct | pass | none | `656eef755e74…` |
  | out-of-bounds write (one element past C) | fail | `sanitizer.memcheck` | `8eb2addcaed5…` |
  | last row of C never written | fail | `anticheat.nan_coverage`, `numeric` (128 unwritten elements per distribution) | `cd1bd70379a2…` |
  | host library calling `cublasGemmEx` (`.so` module, D222) | fail | `anticheat.libraries` (`libcublas.so.13`, `cublasCreate_v2`, `cublasGemmEx`), `anticheat.kernels` (launched `nvjet_sm121_hsh_mma_…`, declared `gemm` not launched) | `bc071da81111…` |

  - Each aggregate references its 8 sub-check records:
    - numeric, sanitizer and anti-cheat at the module's shape;
    - numeric at 2 edge shapes;
    - numeric at 3 hidden shapes.
  - No checker errors occurred. Report `ab7dc41e1a30…` lists the four variants.
- **WP3 gate checks** (`tools/kta_gate/specific_wp3.py`, registered in `specific.py`, D232; unit-tested in `tests/unit/test_gate_wp3.py`). The gate:
  - runs `pytest -m gpu tests/acceptance/WP3` and requires its four tests to pass;
  - requires these records to be listed in the evidence manifest, real and clean: Measurement and Check from a GPU worker, `measure-v1`/`v2`, v2's noise reports with at least 30 groups, the all-as-expected checker selftest, and one significant and one non-significant compare under v2.
- **Acceptance tests** (`tests/acceptance/WP3/`, markers `gpu` and `acceptance`). Each test runs in a fresh temporary store against spark:
  - the checker verdicts;
  - the noise probe (vector add, 30 groups) and the v2 revision;
  - compare: the 1050-step kernel significant and the identical kernel not;
  - profile: a Profile record or NotSupported for missing counter permission.
- **Unit tests** (`tests/unit/test_measure_checks_units.py`):
  - statistics;
  - protocol versions;
  - inputs;
  - the bench job builder;
  - numeric verdicts;
  - parsers for ELF, nsys sqlite, SASS and ncu CSV;
  - hidden shapes.
- **CLI** (D234): `kta measure protocol`, `kta measure noise`, `kta measure compare`, `kta measure profile`, and `kta check selftest`.

## Fixtures and simplifications

- No `# kta:fixture` markers added. The inherited WP1 marker in `src/kta/store/bench.py` is unchanged and unused by WP3.
- Allowed simplification used: kernel tracing uses Nsight Systems (`nsys profile --trace=cuda`) instead of a CUPTI client (D221).
- Deliberately simple kernels (D94, D233). They are real modules built and run on spark:
  - an f32 vector add;
  - a 16×16-tile FP16 GEMM;
  - FMA chains of 1000 and 1050 steps;
  - the three faulty GEMM variants.

  The FMA problem is timing-only (`reference_impl: "none: …"`), so it is never numerically checked.
- Problem, Contract and Case records of these workloads are written by the tool actor `kta measure selftest`. They are not user Cases.

## Known issues

- **No profiles on spark (D228).** `kta measure profile` exits 3 with "GPU performance counters are restricted to administrators (ncu ERR_NVGPUCTRPERM)" (`evidence/…/profile.err`). spark's driver has `RmProfilingAdminOnly=1`. Enabling the counters is a system configuration change on spark (or a privileged container) and needs the user's decision. The Profile path, a CSV parser and Profile record writing, is implemented but has not produced a record.
- **Clocks are not locked.** Locking needs root. Clocks and throttle reasons are recorded at the start and end of every measured phase instead. Between-job drift (up to 0.9% for the GEMM, tracking temperature) is included in the noise floor rather than removed.
- **The torch.matmul baseline is not measured here.** The bench job loads `cubin` and `so` modules. Measuring torch under the same protocol needs a container variant of the bench job; WP7 owns the baseline.
- **Hidden shapes leak through Build records.** A hidden-shape module built through `build_for` gets an ordinary Build record whose `build_key` contains its launch grid and scalars. Check records name only the category, but an agent that can read Builds could infer the shapes. WP6 or WP8 should restrict agent access to checker-built Builds or mark them.
- **Determinism is always required.** Contract has no field to declare nondeterministic kernels (atomics, split-K), so such kernels would fail `anticheat.determinism`.
- **Contract identity needs the Case record.** `full_check` compares the Contract with the one the Case record references. Baseline identity is checked only as "at most one baseline ref", because no baseline records exist yet.
- **SASS checking is unit-tested only.** No delivered run uses SASS expectations: the selftest GEMM declares none. The WP2 CUDA backend writes the `sass` blob that `check_sass` reads.
- **bf16 inputs** raise NotSupported (numpy has no bf16). Only f16, f32, f64 and i32 are generated.
- **Timer resolution.** CUDA event times move in 0.032 µs steps on GB10. Very stable kernels therefore get degenerate bootstrap CIs (both ends equal), as in the compare Reports. The significance rule still requires the 3 × floor margin.
- **The dev store holds earlier construction runs.** It also contains records from the work-in-progress commits `aa83ac2`, `cef1739` and `3390551` (short noise probes, checker runs), and the dirty compare/checker runs at `8320617`. Only the records listed in the evidence manifest are evidence.
- **Data written during WP3:**
  - spark: `~/kta-work/src/<commit>` for the WP3 commits and `~/kta-work/data/{dev/WP3,test-worker}`;
  - control host: `~/kta-data/dev/WP3`.

## New decisions

D220, D221, D222, D223 (needs user review), D224 (needs user review), D225, D226 (needs user review), D227 (needs user review), D228 (needs user review), D229, D230, D231, D232 (needs user review), D233, D234.

## Needed changes

- **Spec changes made in this WP** (spec/modules/measure-check.md, "WP3 implementation notes"):
  - return types with record IDs (D224);
  - `full_check` keywords `case_id`, `shape`, `build_for` (D227);
  - `check_sass(..., *, ctx)` (D229).
- **Additive changes outside measure/checks:**
  - `src/kta/cli/measure_cmds.py` (new) and two lines in `src/kta/cli/main.py` (D234);
  - `tools/kta_gate/specific.py` registers `wp3` from the new `specific_wp3.py`.

  Nothing in `core`, `store`, `execution` or `backends` changed.
- **WP6:**
  - call `full_check`/`verify_space` with a `build_for` that compiles a generated kernel for another shape;
  - keep checker-built Builds of hidden shapes away from agents;
  - use `Checked.checker_error` to park a task instead of counting an agent failure.
- **WP7:** the torch.matmul baseline under `measure-v2` needs a container variant of the bench job (Known issues).
- **User:** decide whether to enable GPU performance counters on spark (D228).

## Gate output

(to be filled in)
