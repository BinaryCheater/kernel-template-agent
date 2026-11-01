# SYSDEBUG-2 morning brief

For: the user, before the supervised round. Written by the assistant on 2026-11-02 (night, UTC 18:00–20:00). Nothing here is a research result or an acceptance.

## 1. Where things run now

| What | Where | Notes |
| --- | --- | --- |
| Trial store `SYSDEBUG2` (empty: no case, no loop, no agent run) | spark-hk, `~/kta/data/dev/SYSDEBUG2` | code `1c02174` (branch `sysdebug/2`), env `~/kta/kta-env-sysdebug2.sh`, services `~/kta/longrun-sd2.sh start|stop` |
| Frontend / API for the trial | spark-hk 4174 / 7312; on the Mac **http://127.0.0.1:14174** (tunnel 14174→4174, 17312→7312) | reopen the tunnel if it dropped: `ssh -N -L 14174:127.0.0.1:4174 -L 17312:127.0.0.1:7312 spark-hk` |
| Rehearsal store `SYSDEBUG2_DRY` (scratch, not evidence) | spark-hk, logs `~/kta/runs/sysdebug2-dry/` | the assistant's dry run of the round, section 4 |
| Old store `SYSDEBUG1` and its services (4173 / 7310) | spark-hk | untouched |
| Mac and slai | not needed for the trial | slai only finished a CPU replay for WP26 |

## 2. What changed since your last look

1. **Review page** (new menu item *Review*, decision D690):
   - *Settings in force*: every setting a run reads, grouped by the step that reads it, with who decided it — **you**, an **assistant default nobody calibrated** (orange), the **design documents** (grey, never confirmed by you) or a **calibration run**. With `?loop=<id>` (the case page links it: "settings this run assumes →") it also shows which settings versions that loop ran under. Below: the hardware numbers the cost model uses, each with measured / documented / default.
   - *Your notes*: everything you marked.
2. **Annotations** (D691): ✓ / ? / ✎ buttons on every setting row, every record drawer (any step's evidence) and every round card. "?" and "✎" ask for one sentence. Close a note with Done / Keep / Later.
3. **Wait before re-measuring** (D692): a timing judged "not quiet" is retried after 30 s instead of immediately (new setting `measure.requeue_wait_s`).
4. **Home flowchart and run cards redrawn** (D693): cards with a colour band for who does the work (grey KTA, green compile/GPU job, violet agent), solid main path coloured up to the current step, branches on one bus, the way back dashed, a glow on the occupied step; runs as cards with a stage bar, round, time in step and controls.
5. **Quiet rule relaxed in SYSDEBUG2** (controls v2, your request; values chosen by the assistant): other-process CPU cores 0.5 → 2.0, in-window SM clock span 0.15 → 0.25; GPU processes, temperature, throttle flags and 3 retries unchanged. Evidence: SYSDEBUG1's 32 rejections came from the co-located API/web (0.5–1.4 cores on a 20-core host) and natural 16–20 % clock span; a deliberate compile (6.9 cores) and 60–90 % clock drops still reject. **The Review page shows this as "set by you" because it was written as the operator; please confirm or change the values.**

## 3. Rehearsal (scratch store SYSDEBUG2_DRY, no agent, nothing tuned)

You can browse it: **http://127.0.0.1:14175** (tunnel 14175→4175; API 7313 on spark-hk). Command logs: `~/kta/runs/sysdebug2-dry/` (`NN.command/.out/.exit`, `step-NN.*`, `steps*.log`).

Setup: `kta init --references`, worker register, target probe, the relaxed quiet thresholds, `kta case add knowledge/seed/cases/gemm_fp16.yaml` (three shapes: 4096³, 8192×1024×4096, 1024×1024×16384), `kta prove family --from-dir …/gemm_fp16_tilelang --target gb10`, then `kta solve` + `kta loop step` until the loop stopped.

| Step | Time | What happened |
| --- | --- | --- |
| Lower bound | <1 s | 1495.0 / 747.5 / 373.8 µs, tensor-bound for all three shapes |
| Find templates | <1 s | the seed family `gemm_fp16_tilelang@1` (7 templates) |
| Pick candidates | <1 s | per shape: 70 feasible of 42.5 M points, exhaustive, 66 kept (198 in all), top 8 per shape chosen for measurement (2 of them random) |
| Compile | 15.4 min | 198 candidates, all built (one at a time, ~4.5 s each) |
| Check | 23.9 min | 440 checks; **all 24 top candidates failed** (below) |
| Measure | blocked, then ~2 min | blocked at once: **no measurement Protocol** (`kta measure protocol` was missing from setup; added, then *Retry* through the API). Then only the **3 cuBLAS baselines** were measured (1483.7 / 767.0 / 441.1 µs); one attempt was rejected (SM clock span 0.91) and the 30 s retry passed |
| Judge | <1 s | gap class *correctness* (rule 1) → task **T6 "fix the family"** → state *waiting* |

### Findings (each needs your eyes)

1. **No candidate can pass the checker.** Of the 24 top candidates, 16 fail `numeric:cancellation` **and** the hidden shape `unit_dimension`; 8 fail `numeric:cancellation`, `extreme` and `large_dynamic_range`.
   - The numeric failures are element-wise (`atol = rtol = 0.01`) on stress distributions: e.g. 4096³ cancellation, 480 of 16.8 M elements off, max error 8.06, while the overall relative L2 error is 2.1e-4 (limit 1e-3). The numbers are nearly identical across candidates, which points to fp16 rounding with K = 4096 rather than a kernel bug. **Question for you: is the element-wise tolerance on these distributions the standard you want, and would cuBLAS itself pass it?**
   - `unit_dimension`: the TileLang build of a hidden shape with a dimension of 1 crashes in code generation (`CodeGenTileLangCUDA`). The case's own shapes are all ≥ 1024. **Question: should hidden shapes for this case include unit dimensions (a scope question), or must the family handle them (a T6 job)?**
   - SYSDEBUG1's GEMM round showed the same "checks passed 0" before it stopped on `JSONDecodeError`.
2. **The lower bound is not a bound.** For 4096³ the bound is 1495.0 µs but cuBLAS measured 1483.7 µs. The bound uses 91.9 TFLOP/s for the tensor pipe; the calibration upstream has a per-SM pipe rate of ~124 TFLOP/s at boost clock and a sustained chip rate of ~95 TFLOP/s. Which one the bound uses is a decision (the "two tensor peaks" item of WP26).
3. **Setup gap:** a fresh store needs `kta measure protocol` before any measurement; nothing warns until the loop blocks in *Measure*. Fix: add it to `kta init` or make the loop say so before building.
4. **Budget:** with default settings one round costs ~40 min before anything is measured (198 compiles + 440 checks). SYSDEBUG1 used your functional-test budgets (shortlist 8, measure 4 points incl. 1 random, solver limit 10 s); the new store does not have them.
5. **A large clock drop on the first measurement after idle** (0.91), as in SYSDEBUG1. The 30 s wait handled it here; whether the protocol's warm-up should absorb it is open.
6. **Inbox silent at S0:** the T6 task was created and waits for release (stage S0 allows no agent), but *Needs you* says "Nothing".
7. **Init warnings:** CUTLASS, FlashAttention-2 and Marlin chains violate `multi_buffer_rotation`'s lifetime rule `P ≤ S − 2 + β` with their own parameters: the rule is stricter than practice, or the bindings are wrong.

## 4. Decisions at the start of the session (about 5 minutes, on the Review page)

1. Confirm or change the relaxed quiet thresholds (section 2, item 5).
2. Round budget for the trial: carry over SYSDEBUG1's functional-test budgets (shortlist 8, 4 measured points, solver 10 s) or keep the defaults (~40 min of compile/check per round).
3. Correctness standard for stress distributions (finding 1) — this decides whether *Measure* sees any candidate.
4. Hidden unit-dimension shapes in scope or not (finding 1).
5. Which tensor rate the lower bound uses (finding 2).

## 5. The supervised round, step by step

Open http://127.0.0.1:14174. Add the case (Cases), then on Home use **Step** for one step at a time. After each step, mark ✓ / ? / ✎ where you look.

| Step | Look at | Judge |
| --- | --- | --- |
| Lower bound | Case page, the three shape cards | is the bound below the library baseline (finding 2)? |
| Find templates | Case page "Built from"; Knowledge → the family's templates | right templates, anything missing |
| Pick candidates | Round card "searched"; Candidates list | search space, what was cut and why, top-k and random share |
| Compile | Round card funnel; Pipeline (time) | failures, time per build |
| Check | Round card funnel; open a failed Check in the drawer | do the failures mean a bug or a standard (finding 1) |
| Measure | Activity (measurement and interference lines); Measurement drawer | accepted vs rejected, reason, noise |
| Judge | Round card result; GapDecomposition drawer; the created task | the class, the route, who gets the work |
| Throughout | Review → Settings in force (`settings this run assumes →` on the case page) | which defaults decided the step |

At the end: Review → Your notes is the list to go through (Done / Keep / Later).
