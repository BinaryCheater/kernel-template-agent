# SYSDEBUG-2 morning brief

For: the user, before the supervised round. Written by the assistant on 2026-11-02 (night, UTC 18:00–20:00). Nothing here is a research result or an acceptance.

## 1. Where things run now

| What | Where | Notes |
| --- | --- | --- |
| Trial store `SYSDEBUG2` (one case added, no loop, no agent run) | spark-hk, `~/kta/data/dev/SYSDEBUG2` | code `c98fcb3` (branch `sysdebug/2`), env `~/kta/kta-env-sysdebug2.sh`, services `~/kta/longrun-sd2.sh start|stop` |
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

Already done in SYSDEBUG2 (setup only, logs `~/kta/runs/sysdebug2/11–13.*`): `kta measure protocol`, `kta case add knowledge/seed/cases/gemm_fp16.yaml` (case `20f53db3…`, three shapes), `kta prove family … --target gb10`. **No loop has started.** The frontend has no start button: the assistant starts it with `kta solve 20f53db3… --steps 1` while you watch (after the decisions in section 4), then you use **Step** on Home (or on the case page) one step at a time. After each step, mark ✓ / ? / ✎ where you look.

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

---

# Update, 2026-11-02 05:30 local: fixes and a full round (SYSDEBUG4)

Frontend http://127.0.0.1:14174 now shows store **SYSDEBUG4** (code `23d71d9`, branch `sysdebug/2`, pushed). SYSDEBUG1–3 and the rehearsal store stay on disk.

## What was fixed (decisions D694–D701)

| Problem seen in the rehearsal | Fix |
| --- | --- |
| Full check (5 distributions, edge and hidden shapes, sanitizer) before every measurement | Tiered: a quick check before measuring (own shape, `normal`, loose level, anti-cheat, SASS); the full check (strict level) only for each shape's fastest measured candidate, in judging |
| Element-wise tolerance against FP64 for an FP16 kernel | Library-relative acceptance: the candidate's rel L2 and max error within ×4 (loose) / ×1.5 (strict) of cuBLAS's own error on the same inputs, plus floors. **Factors are the assistant's proposal** |
| Edge/hidden shapes outside the task (1×1×1, …) | Contracts declare `shape_scope`; the GEMM cases are `static` (their three shapes only); the assistant's edge shapes removed |
| 198 serial compiles | 16 per shape (new knob `solve.compile_count`), 6 at a time |
| Missing measurement Protocol blocked the loop | `kta init` writes it |
| "Agent fixes it … running" while nothing ran | Shown as queued and not running, with the reason; *Needs you* lists the held task |
| Case page could not show any of this | Anomalies at the top, "what each step ran" (jobs per kind, count, time, overlap, blocks, retries, rejected measurements), check failures grouped by reason with numbers, the correctness standard with who wrote it |
| (found in SYSDEBUG3) judging re-ran forever: creating the model task failed because Measurements did not name their candidate, and the error escaped | Measurements carry `candidate`; any error while deciding a transition blocks the loop and is logged |

## Round 1 of SYSDEBUG4 (no agent, nothing tuned)

| Step | Took | Result |
| --- | --- | --- |
| Lower bound → pick candidates | 1 s | 198 shortlisted, 16 per shape compiled + 2 random per shape |
| Compile | 1 min 17 s | 54 built (most from the compile cache) |
| Check before measuring | 3 min 49 s | **24 of 24 passed** (72 check runs, one at a time) |
| Measure | 1 min 35 s | 8 per shape, no attempt rejected |
| Judge | 1 min 43 s | full check of each shape's fastest: 3 of 3 passed (strict); **rule 2: the cost model is off → T4 (fix the model)**, queued, held at S0 |

| Shape | Lower bound | Ours (best) | cuBLAS | Ours vs cuBLAS | Model predicted |
| --- | --- | --- | --- | --- | --- |
| 4096³ | 1495.0 µs | 1985.0 µs | 1496.1 µs | 33 % slower | 1578.3 µs |
| 8192×1024×4096 | 747.5 µs | 780.3 µs | 777.2 µs | 0.4 % slower | 1052.9 µs |
| 1024×1024×16384 | 373.8 µs | 520.9 µs | 433.9 µs | 20 % slower | 562.6 µs |

Median model error per shape: 23 %, 7.7 %, 3.5 %. cuBLAS on 4096³ sits at the lower bound (1496.1 vs 1495.0), so that bound is at best tight; finding 2 still stands.

## Known issues left

- The full checks of this round ran (3 passed) but their mapping was lost when the transition wrote the loop data; fixed for later rounds (`final_checks` is now carried), so this round's page shows only the quick checks.
- The 6 control-arm builds ran one at a time (the page flags it); fixed after the round, not yet seen in a run. Quick checks also run one at a time (72 runs, 3.5 min).
- The loop now waits on T4 (fix the cost model): moving on needs either an agent (stage S1) or a decision about the model.
- Still the user's: tolerance factors, the lower bound's tensor rate, the quiet thresholds (shown as "set by you"), the reference-chain lifetime warnings.

## Update 2026-11-02 04:00 — SYSDEBUG5 (S1, deferred model fixes)

User requests: release S1; show the agent configuration in Settings; pass anything not obviously
wrong; relax the buffer lifetime rule; do not fix the model yet, run the system through.

- Code `7943bb5` (D702–D705 and the final-check fix). Store SYSDEBUG5 on spark-hk; web
  `http://127.0.0.1:14176` (tunnel to 4176), API 17314 → 7314. SYSDEBUG4 (14174) is unchanged.
- Controls v2: quiet thresholds as SYSDEBUG4 v2 (assistant values). v3: `judging.defer_model_fix`
  true and `release.stage` S1, citing the user's words.
- References now import with 0 warnings (multi_buffer_rotation@2).
- Round 1 (no agent): build 63 s, quick checks 230 s (24/24 passed), measuring 329 s, judging
  100 s (3/3 strict full checks passed; the loop data kept only one of them, fixed in `7943bb5`).

| Shape | Ours | cuBLAS | Spearman | Median error |
|---|---|---|---|---|
| 4096³ | 2013.7 µs | 1482.7 µs | 0.83 | 23 % |
| 8192×1024×4096 | 777.2 µs | 768.0 µs | -0.83 | 7 % |
| 1024×1024×16384 | 515.8 µs | 428.8 µs | 0.65 | 4 % |

  The 4096³ rank correlation went from 0.21 (SYSDEBUG4) to 0.83 on the same predictions: with six
  pairs and tied predictions it is noise. cuBLAS again measured below the lower bound (1482.7 vs
  1495.0 µs).
- Route `model_fix_deferred` → T1 "Improve the kernel", held at S1. To run it: Settings → Agent
  prompts → put codex-v3 in force (operator token, reason), then release the task (case page or
  Release stage tab).

## Update 2026-11-02 05:00 — SYSDEBUG7 (codex-v4 in force, T1 ready)

User requests: put the profile in force without asking; more detailed prompts; fix the family
pick; show formulation, verification and the agent's work in the loop.

- Code `9a6d160` (frontend CSS from the next commit, rebuilt in place). Store SYSDEBUG7; web
  `http://127.0.0.1:14178` (tunnel to 4178), API 17316 → 7316. SYSDEBUG6 (same code minus the
  header fixes) was replaced on these ports; its store is kept. SYSDEBUG4 (14174) and
  SYSDEBUG5 (14176) still run.
- codex-v4 put in force by the assistant at the user's word (reason quotes it); batch
  environment `codex-v4-v4` built.
- D706: after an accepted T1/T6 the next round solves the delivered family; D707: routed tasks
  carry the case header (per-shape best / reference / bound / prediction / rank rho, family,
  gap terms, situation) and codex-v4 has a detailed T1 template; D708: the round card shows six
  stages (pick family, formulation, solve + proof gate, compile/check/measure, judge, agent
  task) with who does each.
- Round 1: quick 24/24, strict 3/3 (all three kept now). Best 2015 / 774 / 515 µs against
  cuBLAS 1490 / 770 / 424 µs. Proof gate kept 64/64 per shape. Route `model_fix_deferred` → T1,
  held at S1 until released.
