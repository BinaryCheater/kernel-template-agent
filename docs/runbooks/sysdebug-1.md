# System debug run 1 (SYSDEBUG-1): first end-to-end run on spark-hk

**For:** an AI operator (Codex) driving KTA on behalf of the user. **Goal:** the first real end-to-end run of the system, in a few hours, with as little human attention as possible, ending in one written report the user can read in ten minutes.

This is a debugging run, not an experiment. Its value is in finding out where the system breaks and recording real numbers. Nothing here is a research result, and passing a phase is not user acceptance.

---

## 0. Rules for the operator

1. **Read first:** `AGENTS.md`, `spec/conventions.md` §5–§7 and §10, `docs/design/11-processes-resources-concurrency.md` §1.1, and the README "How a case is solved". Skim the rest only when a step fails.
2. **Where it runs.** Everything runs on **spark-hk** in long-run mode (`KTA_MODE=spark-hk`): control plane, agent CLIs, CPU and GPU jobs. The Mac is only for the browser, through the existing tunnel `http://127.0.0.1:14173` (frontend) and `127.0.0.1:17310` (API).
   - Why not locally: every job would cross the slow hk-relay link (about 1 MB/s, 2.6 s per new ssh connection), and Mac-side agents would need the Clash proxy.
   - Run long commands on spark-hk inside `tmux` (session `sysdebug`), so a dropped ssh connection does not kill them.
3. **Time boxes.** Each phase has one. When a phase fails:
   - at most **two fix attempts, 20 minutes in total**, and only if the fix is local and obvious (a config value, a missing record, a one-line bug);
   - otherwise record the failure with its first error and move to the next phase that does not depend on it.
   - Never spend more than 30 minutes stuck in one place.
4. **Code fixes** go on branch `sysdebug/1` in the KTA repo, with commit format `SYSDEBUG-1: <summary>`. Each fix is listed in the report. No push, no merge, and no changes to spec interfaces (list them under "needed changes" instead).
5. **Forbidden:**
   - sudo or system configuration changes on any machine;
   - deleting or overwriting existing data directories (always create a new one);
   - fixtures or fake results presented as real (conventions §10);
   - editing records by hand;
   - spending beyond the agent budget in §6.
6. **Credentials.** Never print, copy into logs or commit tokens or `auth.json` contents. The credential step (P0.6) is done by the user, or by you only with the user's explicit go-ahead in chat.
7. **Record as you go.** Write each phase's outcome into the report (§8) as soon as the phase ends, not at the end of the run. Record IDs, not prose, are the evidence.

---

## 1. What the run should show

| Phase | Question | Agent spend | Time box |
| --- | --- | --- | --- |
| P0 | Is spark-hk at the current main commit, with a fresh store, and are the API and frontend up? | none | 40 min |
| P1 | Do the infrastructure self-tests pass on the real GPU, and what is the noise band? | none | 30 min |
| P2 | **S0 golden path:** does one FP16 GEMM case go solve → build → check → measure → gap decomposition, with no agent? | none | 60 min |
| P3 | **S1, one hand-released agent step:** does one agent task run, get checked and get recorded, on the subscription login? | 1–2 tasks | 45 min |
| P4 | **Learning, analysis only:** does one small reference go through decompose (T2) → admit? | 1–2 tasks | 45 min |
| P5 | **S2-lite:** does a case loop route its gap to an agent task automatically, within a small budget? | ≤ 3 tasks | 45 min |
| P6 | Write the report | none | 20 min |

P1 and P2 are the core. If P2 cannot reach a measured kernel, still try P3 and P4: they do not depend on it.

---

## 2. P0 — Deploy the current main to spark-hk (no agent)

Facts as of 2026-10-31:
- spark-hk runs WP23's long-run deployment at commit `4b95c5c`:
  - code in `~/kta/src/<commit>`; environment in `~/kta/kta-env.sh` (it sets `KTA_SRC` and `KTA_WP=WP23`);
  - services via `~/kta/longrun.sh start|stop|status`: the API on 7310, the web preview on 127.0.0.1:4173 and on the Tailscale IP;
  - the store in `~/kta/data/dev/WP23`, which is fresh and holds no user data.
- GitHub `main` is now `0c6dc48` (WP22–WP24 merged).
- The docker group is active, and the image `kta-worker:tilelang-0.1.15` is built on spark-hk.

Steps:
1. **Get the code.** Clone `main` from GitHub on spark-hk into `~/kta/src/<commit>` (the repo is public), or use a git bundle from the Mac as WP23 did. Then `uv sync --frozen` and `npm ci && npm run build` in `web/`, with the caches under `~/kta` (see `kta-env.sh`).
2. **Environment.** Write `~/kta/kta-env-sysdebug.sh`: a copy of `kta-env.sh` with `KTA_SRC` pointing at the new commit and **`KTA_WP=SYSDEBUG1`**, so the store is the new `dev/SYSDEBUG1`. Leave the old file and the old store untouched.
3. **Services.**
   - Stop the old services (`longrun.sh stop`; the old store holds nothing the user needs).
   - Point `longrun.sh` at the new env file and start it.
   - Check `curl 127.0.0.1:7310/meta`, and the Mac's `http://127.0.0.1:14173`.
4. **Store and worker.**
   - `kta init`
   - `kta worker register spark-hk`. The labels must include `backend.tilelang=0.1.15` and `gpu=sm_121`.
   - `kta target probe`
   - `kta doctor`
5. **Operator token.**
   - `kta controls token`; the file lands in `~/.config/kta/operator-token`, so do not print it.
   - Every `POST /controls` and `POST /release` below sends it as the header `X-KTA-Operator-Token`.
6. **Credentials (user step).** KTA links `auth.json` from `~/.config/kta/auth/codex/` into each work unit's isolated CLI home. spark-hk already has a Codex login in `~/.codex/auth.json`. Ask the user which of two options to use:
   - **(a) share the login:** `mkdir -p -m 700 ~/.config/kta/auth/codex && ln -s ~/.codex/auth.json ~/.config/kta/auth/codex/auth.json`;
   - **(b) a separate copy:** `cp` instead of `ln -s`, then `chmod 600`. Use this if token-refresh conflicts with the user's interactive Codex show up.

   Then run the Settings → System → Test for codex. It should pass by direct TLS, with proxy policy `off`.
7. **Pass:** `kta doctor` is OK, the worker is healthy with TileLang, the API and frontend are reachable from the Mac, and the codex connectivity test passes.

---

## 3. P1 — Infrastructure self-tests on the GPU (no agent)

Run each of these and record its output and record IDs:
1. `kta backend smoke` (vector add) and `kta backend selftest` (executor behaviour).
2. `kta measure protocol` (writes or finds the Protocol record).
3. `kta check selftest`: one correct GEMM plus three faulty variants. All three faults must be caught.
4. `kta measure noise --kernel all`. This is the first data for **X6, the noise band**. Record the CV per kernel, the SM clocks and the quiet-host verdicts (WP23). If a measurement is marked `interfered`, record what interfered.
5. `kta prove all` and `kta prove canary`. These are CPU-only; the canary must be caught.
6. Look at the Pipeline page and confirm the GPU-exclusive windows and host-lock waits appear.

**Pass:** all of the above succeed, and the noise CV is recorded. If the noise is above 3 %, write that down: every later comparison depends on it.

---

## 4. P2 — S0 golden path: one FP16 GEMM case, no agent

The release stage of a fresh store is S0 (no agent calls). Do not change it in this phase.

Known prerequisites, from WP6, WP7 and WP11:
- a measurement Protocol (P1.2);
- a measured HwModel for gb10;
- the GEMM family's Chain and Family records;
- the case itself.

The construction tool `PYTHONPATH=tools uv run python -m kta_wp7 setup`, followed by `... peaks`, wrote these in WP7. If it insists on `KTA_WP=WP7`, look for the equivalent `kta` commands first: `kta calibrate hw` for peaks, `kta prove family --from-dir knowledge/seed/families/gemm_fp16_tilelang --target gb10` for the Chain and Family records. Record which path worked.

1. `kta case add gemm_fp16`, the seed case: one square and two non-square shapes plus edge shapes.
2. `kta calibrate hw`, or the `kta_wp7 peaks` step. This also gives the first **X10** numbers, the GB10 peaks. Record them.
3. `kta solve <case-id> --steps 1` repeatedly, or `kta process start solve --ticks N` in tmux. After each step, `kta loop show <loop>` shows the state.
4. Follow it through to the end:
   - SOL bound;
   - chain retrieval;
   - solve, with its shortlist;
   - candidate proof gate (WP24; expect `claims: exploration` for the seed family);
   - build;
   - SASS intent check;
   - correctness on the top-k;
   - measurement under the exclusive host lock, with a quiet verdict;
   - gap decomposition;
   - judgement.
5. Record:
   - the best measured time per shape;
   - the torch / cuBLAS baseline time;
   - T_SOL;
   - each gap term (ε, δ, Δpre, L_rep, irreducible), or why it is `not_assessed`;
   - the routing decision;
   - the selection-arm results (model top-k vs random vs copied);
   - wall-clock time per stage, from the Pipeline page.

**Pass:** at least one correct kernel is measured for at least one shape, and a GapDecomposition record exists. It does not matter whether the kernel is fast.

---

## 5. P3 — S1: one hand-released agent step

1. **Agent profile:**
   - `kta profile write codex-v2`
   - `kta profile approve --reason "SYSDEBUG-1" <profile-id>`
   - `kta agents env build sysdebug1 --profile <profile-id>`

   The model is `gpt-5.4` at effort `medium`; leave it as is.
2. **Move the store to S1** with `POST /controls`, body `{"expected": <current version id from GET /controls>, "changes": {"release.stage": "S1", "budget.agent_instances": {"start": 1, "max": 1}}, "reason": "SYSDEBUG-1 P3"}`. You can also use the Settings page.
3. **Create one small task.** Pick the cheapest one the system offers, preferring this order:
   - if P2 routed a T4 or T6, use that task;
   - otherwise `kta task create T5 --goal "<one microbenchmark the hardware model lacks, from the inbox or the P2 gap>"`.
4. **Start one runner slot** in tmux: `kta runner start --slot 1 --batch sysdebug1`. In S1, every agent call waits for a release.
5. **Release it.** Read the task card, the header and the plan on the Release page, then `POST /release` with `{"task": "<task id>", "reason": "SYSDEBUG-1 read card"}`. You may release it yourself: the user delegated this for SYSDEBUG-1.
6. Watch the Workflow, Task and Inbox pages. Record:
   - AgentRun ID, session ID, wall time, token usage if recorded, `proxy.used` (expected false);
   - done-check results;
   - retries, and whether a rejection went back to the same session;
   - the work unit name. It should be `<loop>:<role>` (WP23).
7. **Pass:** one AgentRun is recorded end to end (accepted or rejected with a reason), and its workspace, session archive and checks are inspectable.

---

## 6. P4 — Learning, analysis only (T2 → admit)

**Who does reproduction.** Reproduction is a deterministic stage of the learn process: build our Family for the reference's chain, then measure it against the reference baseline within ε. Agents only fill gaps:
- T2 decomposes the reference into a chain;
- T3 writes a missing Template;
- T1 (the kernel role) writes or extends the family generator code the reproduction needs.

In the current code, the reproduce and ablate stages of a runnable reference still raise `NotSupported` (they are dry-run plans, D569). So this run tests analysis only: L0, intake planning, decompose, admit.

1. Pick the smallest reference: `vllm_rms_norm` from the seed (`kta init --references` imports the seed decompositions; or `kta reference add` with its YAML).
2. `kta learn add <reference>`, then `kta expand <reference> --analysis-only "SYSDEBUG-1: reproduce not wired"`. This creates a T2.
3. Release the T2 as in P3. Observe the admission done-check:
   - errors and unexplained warnings go back to the same session (≤ 2 retries);
   - language gaps are counted, not refused.
4. Record:
   - the admission Report;
   - the number of retries;
   - Gaps created;
   - any T3 created (do not release more than one);
   - the learning-plan stages on the Mechanisms page.
5. **Pass:** the T2 ends accepted or escalated with a readable reason, and the admission Report exists.

---

## 7. P5 — S2-lite: one automatic case loop

Only if P2 and P3 passed.
1. `POST /controls`: set `release.stage` to `S2` and keep `budget.agent_instances` at 1. Also lower `budget.rounds_per_case_loop` to 3.
2. Create the BF16 case (`kta case add gemm_bf16`). Run `kta process start solve` and `kta runner start --slot 1` in tmux.
3. **Hard stop:** after 3 agent tasks, or 45 minutes, whichever comes first. Stop with `kta process stop solve` and `kta runner stop`.
4. Record:
   - each round's gap class and the route taken;
   - which task types were created;
   - acceptances and rejections;
   - two-null-round closes;
   - escalations to the inbox;
   - stage-queue depths and waits (Pipeline);
   - total agent time and GPU time.

### Agent budget for the whole run

At most **6 agent tasks**, at most **2 hours of total agent wall time**, and no more than 1 concurrent agent. pi, claude_code and the deepseek harness are **not usable yet**: their adapters raise `NotSupported` (only the Codex adapter exists; see §9).

---

## 8. P6 — The report (the only thing the user must read)

Write `deliveries/SYSDEBUG-1.md` (English) and commit it on `sysdebug/1`. Keep the first screen short.

```
# SYSDEBUG-1 report
## Verdict (5 lines max)
What works end to end now; the first thing that blocks; what the user must decide.
## Phase table
| Phase | Pass/Fail/Skipped | Time spent | Key record IDs | First error (if any) |
## Numbers measured (real, spark-hk, commit <sha>)
noise CV per kernel (X6 input); GB10 peaks (X10 input); per shape: best kernel time, baseline time,
T_SOL, gap terms; agent: tasks, wall time, tokens if recorded, retries; stage wall-clock times.
## Failures and fixes
For each: symptom, root cause (or best guess, labelled), fix commit or "not fixed", how to reproduce.
## Decisions for the user (each one question, options, recommendation, cost of changing later)
## Observations about the design (what felt wrong, slow or unclear in the flow) — max 10 bullets
## What was not exercised and why
```

Also `deliveries/SYSDEBUG-1.evidence.json`: record IDs per phase, commit, data directory, and the worker EnvManifest ID.

---

## 9. Known gaps to expect (do not try to fix them in this run)

- **Only the Codex adapter is implemented.** pi (with the user's DeepSeek v4.1 flash configuration under `~/.pi/agent/` on spark-hk), claude_code and the deepseek harness raise `NotSupported`. Model-choice experiments (X1) that include a cheap tier need the pi adapter first: list it as a needed change if it matters.
- **Model and effort are per profile**, not per task type (D594).
- **Reproduce and ablate are dry-run plans** for runnable references (D569).
- **The seed GEMM Templates do not reach a proof level** (WP24, D643). Candidates are therefore `exploration`, and that is expected.
- **The eager torch baseline records no host readings** (D605). Its timings are unverified.
- **No periodic backup**: take one at the end with `kta store backup ~/kta/backups/<ts>`.
