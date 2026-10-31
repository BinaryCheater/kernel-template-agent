# System debug and trial run (SYSDEBUG-1)

**For:** the AI operator (Codex) that drives KTA, the AI reviewer that checks its work, and the user who decides.
**Goal:** get KTA running for real in three phases:
1. unattended calibration;
2. joint debugging under the user's supervision, until the user understands the system and accepts the method;
3. automatic runs that the user accepts or rejects on data about stable, continuous operation.

Nothing here is a research result. Passing a check is not user acceptance: only the user's decision on a card is.

---

## 0. Roles, cards and rules

### Roles

| Role | Who | Does | Never does |
| --- | --- | --- | --- |
| Operator | Codex, in a tmux session on spark-hk | runs the steps, fixes small local faults, writes a card per decision point | decides a C-level point; edits records by hand; spends beyond the budget |
| Reviewer | a separate AI session that did not run the step (another Codex session, or Claude) | checks each card against the records: data complete, rule applied as written, numbers traceable to record IDs, nothing presented as real that is not | runs experiments; changes the data |
| User | — | decides B and C cards, supervises phase 2, accepts phase 3 | reads raw logs (the cards point to them when needed) |

### Decision levels

- **A, agent decides:** the operator applies the card's rule and the reviewer confirms. The value takes effect and appears as one line in the phase summary table.
- **B, user glances:** the reviewer's recommendation takes effect unless the user vetoes it.
- **C, convince the user:** nothing takes effect until the user decides. The card must contain the data, a chart link in the frontend, and the alternative.

### Card format

One screen, in `deliveries/sysdebug/cards/<id>.md`:

```
# <ID> <question>                         level A|B|C   status proposed|reviewed|decided
Data:      what was measured, n, where (record IDs, frontend link)
Rule:      the decision rule fixed before the run (from this runbook)
Result:    the value / verdict the rule gives
If wrong:  what goes wrong downstream if this value is off
Reviewer:  agrees / disagrees, and why (one or two lines)
Decision:  (user, for B/C) value, date
```

The phase summary is a table of cards, one line each. The user reads summaries and B/C cards, nothing else.

### Rules for the operator

1. **Read first:** `AGENTS.md`, `spec/conventions.md` §5–§7 and §10, `docs/design/11-processes-resources-concurrency.md` §1.1, and the README sections "How a case is solved" and "How the system learns".
2. **Where it runs.** Everything runs on spark-hk in long-run mode (`KTA_MODE=spark-hk`): control plane, agent CLIs, CPU and GPU jobs. The user watches from the Mac through the existing tunnel: frontend `http://127.0.0.1:14173`, API `127.0.0.1:17310`. Run long commands in tmux.
3. **When a step fails:** at most two fix attempts, 20 minutes in total, and only if the fix is local and obvious. Otherwise write the failure (symptom, first error, record IDs) on the phase's card and continue with the next independent step.
4. **Code fixes** go on branch `sysdebug/1`, with commit format `SYSDEBUG-1: <summary>`. No push, no merge, no spec interface changes (list those as needed changes).
5. **Forbidden:** sudo or system changes; deleting or overwriting data directories; fixtures or fake results presented as real; editing records by hand; printing or committing credentials.

---

## Functional-debug priority (user clarification, 2026-10-31)

Before case quality or parameter optimisation, rehearse system functions: a normal stage path,
expected compile/check failure, timeout, pause/resume, partial-progress recovery, correct routing,
and truthful frontend state with inspectable evidence. A workload is only a bounded test input.
A numerical failure is useful evidence for the rejection path; it does not require kernel tuning.

Current Spark HK functional-test controls: shortlist 8, 4 measured points including 1 random,
solver limit 10 s, release S0. These are reversible debugging budgets, not X8/X9 results.
Keep statistical significance, worthwhile improvement, numeric correctness and reproduction
margins distinct. Uncalibrated values stay labelled; functional inspection can proceed before
performance-acceptance calibration is complete. Discuss findings directly with the user;
retain machine-readable evidence without requiring new decision cards for routine repairs.

## 1. Phase 1 — Calibration (unattended)

**Purpose:** a trustworthy ruler and price list before anything is judged. **User attention:** read one summary table at the end.

### 1.1 Setup steps

These steps make no decisions.

1. **Deploy the current `main` to spark-hk** (as WP23 did):
   - code in `~/kta/src/<commit>`, with `uv sync --frozen` and a web build;
   - a new env file `~/kta/kta-env-sysdebug.sh` with `KTA_WP=SYSDEBUG1`, so the store is the new `dev/SYSDEBUG1`;
   - point `longrun.sh` at it.

   The old store `dev/WP23` holds nothing the user needs. Leave it on disk.
2. **Initialise:**
   - `kta init --references`
   - `kta worker register spark-hk`. The labels must include `backend.tilelang` and `gpu=sm_121`.
   - `kta target probe`
   - `kta doctor`
   - `kta controls token`; never print it.
3. **Credentials (user step).** Link or copy `~/.codex/auth.json` into `~/.config/kta/auth/codex/`. Ask the user which: a link shares the login with their interactive Codex; a copy avoids token-refresh conflicts. Then run Settings → System → Test for codex (direct, proxy `off`).
4. **Self-tests:**
   - `kta backend smoke`
   - `kta backend selftest lease`
   - `kta backend selftest timeout`
   - `kta backend selftest cpu-compile --arch aarch64`
   - `kta measure protocol`
   - `kta check selftest`: all three planted faults must be caught.
   - `kta prove all`
   - `kta prove canary`: the canary must be caught.

### 1.2 Decision points

| Card | Question | Data | Rule | If wrong | Level |
| --- | --- | --- | --- | --- | --- |
| **C1-X6** | How big a difference is a real improvement (noise band)? | `kta measure noise --kernel all`, repeated with agents idle and with one agent CLI active; CV per kernel, clocks, quiet verdicts | protocol floor = max CV of group medians rounded up to 0.1 percentage point; significance requires disjoint 95% CIs and a difference above 3 × floor. Report rejected attempts as well as accepted timings; the policy improvement band is a separate setting | too narrow: noise accepted as gains; too wide: real gains ignored and cases close early | A |
| **C1-X10** | What are GB10's peaks and pairwise interferences? | `kta calibrate hw` (peaks and microbenchmarks), interference pairs A, B, A+B | values with their spread go into a new HwModel version; a peak whose spread exceeds the band is flagged | SOL too optimistic: gaps never close; too pessimistic: cases stop early | A |
| **C1-QUIET** | When is the host quiet enough to measure? | host snapshots around every measurement above; one deliberate concurrent compile during a window | thresholds = idle distribution + margin; the deliberate compile must be flagged `interfered` | unexplained timing noise, or good measurements thrown away | A |
| **C1-MEM** | How much memory may concurrent CPU jobs use on the unified-memory host? | peak RSS of the compile and check jobs above, GPU memory during measurement | cap = total − measurement peak − margin | jobs killed, or the GPU starved during measurement | A |
| **C1-X11** | When do the MIP and the L2 simulator agree enough? | `kta`'s MIP ↔ L2 consistency check on the GEMM family; distribution of Spearman ρ and of bound violations over sampled subspaces | threshold below the observed distribution's lower tail; bound violations must stay at 0 on a correct table | a wrong model passes unnoticed, or false alarms on every run | A |

**Phase 1 summary:** `deliveries/sysdebug/phase1.md`. It holds the five card lines, the self-test outcomes and any failures. **Exit:** all self-tests pass and the five cards are reviewed. If C1-X6 or C1-X10 cannot be produced, stop and report: phase 2 needs both.

---

## 2. Phase 2 — Joint debugging (the user supervises)

**Purpose:** one real round of everything, slowly and visibly. By the end, the user understands the system and accepts or corrects the method.

**User attention:**
- the user is present for steps 2.2 and 2.3 (the agent calls are released one at a time);
- a method walkthrough at the end;
- four C cards.

### 2.1 S0: one case, no agent

1. `kta case add gemm_fp16`. Build the HwModel from C1-X10. Build the GEMM family's Chain and Family records, with `kta prove family --from-dir knowledge/seed/families/gemm_fp16_tilelang --target gb10` or the WP7 setup tool (record which).
2. Start once with `kta solve <case> --steps 1`, retain its loop ID, then repeat `kta loop step <loop-id>` until the loop judges. Repeating `kta solve` creates a new loop each time.
3. Check each stage on the frontend:
   - SOL;
   - chain retrieval;
   - solve and its shortlist;
   - candidate proof gate (expect `exploration`);
   - build;
   - SASS intent;
   - correctness;
   - measurement under the host lock;
   - gap decomposition;
   - routing.
4. **Pass:** a correct kernel is measured for at least one shape, and a GapDecomposition exists. It does not matter whether the kernel is fast.

### 2.2 S1: agent calls released one at a time

1. **Profile:**
   - `kta profile write codex-v2`
   - `kta profile approve <profile-id> --reason "SYSDEBUG-1 supervised trial"`
   - `kta agents env build sysdebug1 --profile <id>`
2. **Stage.** `POST /controls`: set `release.stage` to `S1` and `budget.agent_instances` to `{start: 1, max: 1}`.
3. Start `kta runner start --slot 1 --batch sysdebug1`.
4. **Release one task of each role.** The user reads the card, the header and the plan on the Release page and releases it:
   - **kernel**: the T1 or T6 that S0 routed, or a hand-made `kta task create T6` for a failing shape;
   - **model**: a T4 on S0's largest prediction error, or a T5 for a missing hardware fact;
   - **knowledge**: step 2.3.
5. For each task, record:
   - AgentRun, session and work unit (expect `<loop>:<role>`);
   - wall time and tokens, if recorded;
   - done checks;
   - retries, and whether each went back to the same session;
   - what the agent pulled (pull log).

### 2.3 Knowledge intake, analysis only

1. `kta learn add vllm_rms_norm`, then `kta expand <ref> --analysis-only "SYSDEBUG-1"`. This creates a T2; release it.
2. Record:
   - the admission Report;
   - retries;
   - Gaps created;
   - at most one T3, released.
3. Reproduction is a deterministic stage of the learn process: our family against the reference baseline, within ε. Agents only fill gaps: T2 decomposes, T3 adds Templates, T1 writes the family code. Reproduce and ablate are still dry-run plans for runnable references (D569). Record this as a gap; do not work around it.

### 2.4 Decision points

| Card | Question | Data | Rule | If wrong | Level |
| --- | --- | --- | --- | --- | --- |
| **C2-O3** | Does the model rank candidates well enough to be worth using? | predicted vs measured on every measured point of 2.1, plus the selection arms (model top-k vs random vs copied) | ρ ≥ 0.5 and model top-k beats random → model guides search; otherwise k and the exploration share go up (measurement-led search) and model work is prioritised | the solver is noise and GPU time is wasted, or a good model is ignored | **C** |
| **C2-O5** | Does gap attribution name the right cause? | fault injection on recorded inputs: degrade the model, delete a Template, cap the solver; the canaries | each injected fault routes to its class; a miss is a blocking defect | model faults sent to kernel writers, knowledge gaps sent to tuning: agent time wasted | **C** |
| **C2-X8** | How many points to measure per round, and what exploration share? | hit rate of the best point vs k, from 2.1 | the smallest k that reaches the best point in all observed buckets, plus the exploration share given by C2-O3 | best points missed, or GPU time wasted | A |
| **C2-METHOD** | Does the system work the way the user expects? | the method walkthrough (below) | the user's judgement | everything later builds on a wrong method | **C** |
| **C2-S2** | May the system run on its own (S1 → S2)? | in 2.2, could the user predict each agent's behaviour from its card and the access discipline? | the user's judgement | loss of control, or needless hand-holding | **C** |

### 2.5 Method walkthrough

This is the checkpoint before phase 3.

The reviewer writes `deliveries/sysdebug/method.md`: for the S0 case and each released task, what happened at each step, why (the record that caused it), and what it produced. Two pages at most, each step with a frontend link.

The coordinator walks the user through it on the frontend. The user decides C2-METHOD. If the method is wrong, change it here, before any tuning.

**Phase 2 summary:** `deliveries/sysdebug/phase2.md`: card lines, failures and fixes, open questions. **Exit:** C2-METHOD and C2-S2 decided by the user.

---

## 3. Phase 3 — Automatic runs and acceptance

**Purpose:**
- the system runs on its own;
- three loops turn in parallel;
- parameters are tuned from logs;
- the user accepts or rejects continuous, stable operation on data.

**Prerequisites:** these do not exist yet; build them in a separate work package, in parallel with phases 1–2.
1. An **output ledger page**, plus a **batch digest**: one screen per batch saying what was learned, how the model moved, which formulation changed, which agent configuration changed, and where each case's performance stands.
2. **Model-trust adaptive measurement:** k and the exploration share follow C2-O3's rule automatically, round by round.
3. **The reviewer as a process:** it reads batch logs, writes cards, and proposes skill and prompt changes as draft profile versions (WP22 workflow; the user approves).
4. **The pi adapter**, with the user's DeepSeek v4.1 flash configuration on spark-hk, so X1 can include a cheap tier.

### 3.1 The three loops

None of them blocks another; they meet only through records.

| Loop | Validated by | Produces |
| --- | --- | --- |
| Knowledge | measurement only (compiles, correct, reproduces within ε) | Templates, chains and families, Findings, couplings, proofs |
| Model | every measurement is a predicted-vs-measured point | HwModel versions, calibration set, scoreboard deltas |
| Formulation | ranking quality of each plan | FormulationPlans; the ones that rank better become family defaults |

An inaccurate model costs extra measurements; it never stops knowledge work or changes what counts as correct.

### 3.2 Run

1. Begin with `release.stage` S2 for case-loop supervision. S2 excludes learning and summarization; move to S3 only after its release criteria are accepted before claiming three concurrent loops. Agent instances start at 1 and rise to 2 only after controlled comparison of accepted throughput, queue age, memory and quiet-window disruption.
2. A handful of cases: GEMM FP16 and BF16 shapes, then one other kernel class once the GEMM batch is clean.
3. References keep flowing through analysis and admission.
4. Batches of about 2 hours, each ending in a digest and the cards below.

### 3.3 Decision points

| Card | Question | Data | Rule | If wrong | Level |
| --- | --- | --- | --- | --- | --- |
| **C3-X5** | How many retries per task? | marginal acceptance of retry 2 vs 3 | keep the retry only while it adds accepted tasks per agent-hour | stuck tasks, or tasks killed that would have passed | A |
| **C3-X9** | Task wall-clock budgets per type | completion-time distribution per task type | p90 + margin | runaway tasks, or premature kills | A |
| **C3-X12** | Concurrent agents and stage capacities | throughput, GPU queue wait, rate-limit hits, stage depths | the setting with the best accepted throughput per agent-hour within the declared queue, memory and error budgets; GPU idle is diagnostic, not an acceptance target | GPU idles, or work goes stale in queues | A |
| **C3-X2** | Reasoning effort | two effort levels on the same tasks | the lower one unless acceptance drops beyond the band | money wasted, or quality lost | B |
| **C3-X3** | Resume scope and work-unit rule | tokens per accepted task, acceptance, model-blame share (3 arms) | the cheapest arm with no acceptance loss and no blame skew | context blow-ups, or biased diagnoses | B |
| **C3-X4** | How agents get knowledge (discipline vs dump vs browse) | acceptance and tokens per arm | the best acceptance per token | agents starved or flooded | B |
| **C3-X7** | Reproduction tolerance ε | reproduction error over the references (once reproduce is wired) | separates known-correct from planted-wrong decompositions | wrong decompositions admitted as knowledge | B |
| **C3-X1** | Which model per task type, including a cheap tier | the same tasks per model: acceptance, cost per accepted task | the cheapest model within the band of the best, per task type | cost ten times too high, or quality collapses | **C** |
| **C3-KNOW** | Is accumulated knowledge usable? | leave-one-out: learn from 6 references, blind-solve the 7th's case, with and without the pool | the pool must beat no-pool beyond the band | the knowledge base grows but does not help | **C** |
| **C3-ACCEPT** | When is a case done, and when does it escalate? | case closings vs remaining gap, escalations per batch | the user's judgement on the observed trade-off | stopping too early (performance lost) or too late (money burned) | **C** |

### 3.4 Acceptance

This is the user's decision, on data from at least three consecutive batches:

- **Continuity:** hours of unattended operation, and interruptions with their causes.
- **Throughput:** tasks per hour, GPU utilisation, queue waits per stage.
- **Cost:** agent time and tokens per accepted task, and GPU time per measured point.
- **Progress:** the gap-closed share of the case set (gap within the band of SOL, or explained) over time, and outputs per batch from the ledger.
- **Control:** escalations per batch (should fall), budget overruns (should be zero), canaries missed (zero).

**Phase 3 summary:** `deliveries/sysdebug/phase3.md`: the acceptance data above, card lines, and the reviewer's verdict. The user decides whether the system may run continuously.

---

## 4. Known gaps

Do not fix these during the run; record them where they bite.

- **Only the Codex adapter exists.** pi, claude_code and the deepseek harness raise `NotSupported`. The pi adapter is a phase-3 prerequisite.
- **Model and effort are per profile**, not per task type (D594).
- **Reproduce and ablate are dry-run plans** for runnable references (D569). C3-X7 waits for them.
- **The seed GEMM Templates reach no proof level** (WP24, D643). Candidates are `exploration`; that is expected.
- **The eager torch baseline records no host readings** (D605).
- **No periodic backup.** Take one at the end of each phase with `kta store backup ~/kta/backups/<ts>` and pull it to the Mac.
