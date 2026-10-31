# SYSDEBUG-1 initial debug checkpoint

Date: 2026-10-31 (Asia/Singapore). Branch `sysdebug/1`.

**Status: initial diagnostics available for supervised visual inspection; Phase 1 BLOCKED.** This is not an accepted S0/S1 transition, completed calibration, or a research result. The runbook explicitly stops before Phase 2 when X6/X10 cannot be produced. The operator followed that boundary.

## What was done

- Read the current runbook, conventions, WP23/WP24 design and delivery boundaries; deployed the code to Spark HK. Latest deployed source: `e7318d54a2b6ef9bd9510357a5b29df74568d517`. API, web preview and the pre-existing Tailscale web binding run in tmux. The Mac tunnel remains `http://127.0.0.1:14173` (web) and `127.0.0.1:17310` (API).
- Used the new `~/kta/data/dev/SYSDEBUG1` store. Old `dev/WP23` and its code were preserved; old launcher saved as `~/kta/longrun-WP23.sh`. New environment is `~/kta/kta-env-sysdebug.sh`; no main branch change, push, model call, login copy/link, sudo or system configuration change occurred.
- Registered `spark-hk` with `gpu=sm_121`, CUDA and TileLang backend labels. Target probing and doctor passed. Both CUDA and TileLang vector adds passed exact checking of 1,048,576 elements. Lease exclusion, timeout recovery and aarch64 CPU compile passed. Evidence entries 1–3 and the original command outputs under `~/kta/runs/sysdebug1/`.
- Fixed GPU container startup: remove the forced unregistered `nvidia` runtime and let Docker satisfy the explicit device request (D645). The real reference-based checker rerun passed the correct GEMM and caught all three planted faults, Report `c907edb357877d2497ca01f7cfa23eef3959b58fa910ecf82323394837f1b530` (evidence 4). Failed pre-fix Check records remain immutable.
- Verified ncu through KTA's ordinary Executor path: Profile `39fb0f72242945e6588af2a26a95eb09973b8b082974d944afe107dff5445c3b`, duration **1,176,224 ns = 1,176.224 µs**, SM throughput **83.19%**. Driver already reports `RmProfilingAdminOnly=0`; no permission change was necessary. Requested DRAM metric is `n/a` in the CSV, not zero (evidence 6). The [NVIDIA permission guidance](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters) was consulted; the real profile, rather than driver configuration alone, establishes access.
- Ran seven proof outlines and the planted proof canary (`3e77e0b8…`, caught). Shared-memory tiling has bounded assurance; six other outlines have no proof level. These are the expected WP24 findings, not seven proved Templates (evidence 5).
- Attempted idle vadd/GEMM noise, hardware calibration and an initial two-pair interference probe. **16 attempts were rejected and retained as real Reports; zero Measurements were accepted.** Initial idle failures are clock-only; later reports also include CPU/GPU activity. No threshold was changed, failed timing accepted, CV fabricated or HwModel calibrated (evidence 7–8).
- Recorded MIP–L2 consistency for 3 shapes × 70 feasible seed configurations: rho 0.5192 / 0.5216 / 0.5361, zero bound violations; L2/MIP up to 2.4898. This is a model-to-model diagnostic using the seed model (evidence 9).
- Fixed measurement CLI controls loading (D646). It previously ignored stored quiet thresholds; it now uses the same stored snapshot as orchestration. Fixed Workflow source navigation under `src/<commit>/src/...` (D647), retaining the existing acceptance assertion. Added targeted regressions for both.
- Clarified System labels: an endpoint applies only to required providers; a failed PATH probe says "not found on PATH", which does not contradict an installed, configured absolute CLI command (D648). No settings or probe behaviour changed.
- Ran the full WP24 **code-stage regression gate** through an Executor CPU job on Spark HK, final source `e7318d5`: **22 pass, 2 warn, 0 fail**, 741 non-GPU tests and 31 contract tests; pyright, import layers and clean frontend build pass. Report `7d37eafb…` (evidence 13; earlier core regression in evidence 10). This gate is regression coverage, not a SYSDEBUG Phase-1 acceptance gate; `scripts/gate` has no SYSDEBUG1 package name.
- The gate's real-Codex-help contract was skipped because the CLI was not on that job's PATH. Ran it separately through the Executor with explicit `KTA_CODEX=~/.local/bin/codex`: **3 passed**, Report `06216b23…`, no model request (evidence 11).
- Applied long-run proxy `off` through the authenticated System API and verified direct Codex TLS preflight. Only the audit action and preflight records were returned; no token or login contents were printed. Research controls and release stage S0 remain unchanged (evidence 12).
- Wrote [seven decision cards and the phase summary](sysdebug/phase1.md), [questions for the user](sysdebug/questions.md), and a [visual walkthrough](sysdebug/walkthrough.md). An independent AI reviewer checked the cards against the read-only API and agreed with their pass/blocked boundaries. The rejected-attempt drawer was visibly verified in the browser.

## Fixtures and simplifications

- No fixture data seeded into SYSDEBUG1 and no new fixture implementation in `src/`. Existing `src/kta/store/bench.py` fixture is unchanged and remains declared by WP1.
- The interference driver plans only two initial pairs, explicitly labelled non-exhaustive; neither completed because the first measured kernel was rejected.
- Initial direct tests from the deployment archive reported 11 failures: the archive lacks git metadata needed by ten provenance tests; the workflow source-path failure was a genuine deployment bug and was fixed. The git-backed final-source code gate passes without relaxing those tests.
- No replacement result, hidden fallback, changed numerical tolerance, weakened checker or new research threshold was used.

## Known issues

- **Measurement policy blocks X6/X10.** `sm_clock_drop_frac = 1 − min(start,end clock)/rated maximum` measures distance below rated maximum, not an observed clock drop. Otherwise quiet attempts are rejected. A validated quiet-clock observable, an idle distribution and explicit margins are needed before retuning; this checkpoint does not choose them.
- C1-MEM lacks compile/check peak RSS, a measured GPU peak and a defined margin. Existing caps remain labelled defaults.
- Profile stores numeric metrics without their CSV units/availability. The real CSV supplies ns for duration and `n/a` for DRAM. Profiler access is verified; full metric coverage is not.
- No completed active-agent X6 arm, accepted Measurement, calibrated HwModel, S0 GapDecomposition or real AgentRun exists. Credentials still await the runbook's user choice of copy vs link.
- Proof counterexamples and unsupported proof levels remain as WP24 expected. Seed GEMM candidates would be exploration; proof execution exit 0 is not assurance.
- Calibration reasons are reachable through Record drawers but are raw JSON; there is no dedicated calibration summary page or `measurement_interfered` Inbox trigger. Cards supply the readable summary for the joint walkthrough.
- Reproduction/ablation and the Phase-3 prerequisites are still separate work. No attempt was made to bypass them.
- The final regression gate keeps WP24's warning about real runs in its own `dev/WP24` namespace. This SYSDEBUG store does not substitute for that acceptance.

## New decisions

D645 (Docker GPU request), D646 (stored measurement controls), D647 (deployment source links), D648 (system diagnostics label scope). All are local reversible implementation repairs; no spec interface or acceptance criterion changed.

## Needed changes

- Joint review of [questions 1–2](sysdebug/questions.md): decide and validate the quiet-clock observable and calibration margins before retrying X6/X10.
- Choose copy vs link for the dedicated Codex login before S1; no credential operation was performed.
- Check one proof counterexample and one rejected timing in the frontend, then review the remaining cards. The user owns any C-level method or stage decision.
- No changes to other module interfaces were made. Corrected runbook CLI syntax for backend self-tests, solve-loop advancement and profile approval.

## Pending real runs

Complete X6 idle/active noise, repeated peaks and interference, deliberate compile interference drill, memory headroom calibration, S0 GEMM and its GapDecomposition, then user-released S1 tasks. None is described as complete here.

## Backup

Final handoff snapshot on Spark HK: `~/kta/backups/SYSDEBUG1-20261031-handoff`, created 2026-10-31T11:25:58Z.

Verified Mac copy: `/Users/dingyi/kta-backups/spark-hk/SYSDEBUG1-20261031/handoff-snapshot`.

- DB sha256: `162261d094bddb7e1bcecf6843e51bc604c0a0a2d01f12a1cb7ad960e1920f71`.
- **527 / 527 object hashes verified.** Every record and blob referenced by the 13 evidence entries is present in this read-only-checked snapshot.
- System configuration included; no login credentials copied. The earlier snapshot is preserved. Base archive and a 655 KB delta were used to transfer the new snapshot without retransmitting unchanged objects; `verified.json` records the check.
- The remote store and services remain live. No periodic backup was scheduled.

## Gate output

Final-source run on Spark HK via Executor, Report `7d37eafbdf353f5ebc2b79397259cf49a1f918ee21fe545a317432afb971c57f`:

```
gate WP24 --stage code: PASSED (22 pass, 2 warn, 0 fail)
  PASS ruff format
  PASS ruff check
  PASS import layers
  PASS residue
  PASS test doubles
  PASS error handling
  PASS print
  PASS subprocess outside Executor
  PASS secrets
  PASS acceptance lock — 7 locked files unchanged
  PASS pyright
  PASS unit tests — 741 passed
  PASS contract tests — 31 passed
  PASS fake coverage — 2 fake(s) covered
  PASS registry (kta doctor, dev)
  PASS delivery note
  PASS fixture declarations — 1 declared
  PASS decision log — 486 decisions
  WARN skipped tests — not explained in the delivery note
  PASS evidence — 5 claim(s)
  PASS WP24 tests — 16 named tests passed
  PASS WP24 spec, seed and pages — spec, z3 lock, seed outlines, pages
  PASS web build — tsc + vite build of HEAD in a clean directory
  WARN WP24 real checker runs — pending real run (code stage): ProofSummary with a level and a caught proof_canary Report in dev/WP24 (clean commit)
```

Skipped test: `tests.contract.test_codex_cli_contract::test_real_cli_lists_every_adapter_option` (CLI absent from the gate job's PATH). Separate explicit-path help-only contract run passed 3 tests. The other warning is WP24's own pending real acceptance namespace, not a hidden SYSDEBUG pass.

## Clock-rule repair (D649; later follow-up)

User authorised autonomous mechanism repair. Design check: (1) verify observed operating stability; (2) support reproducible wall-time comparison; (3) evidence is stable 2515 MHz versus rated 3003 MHz, with no throttle flags; sampling every 50 ms is still a default; (4) applies to event-timed worker jobs, with no claim of continuous in-kernel observation; (5) can miss changes between samples or perturb launch gaps; (6) raw samples, rule version, clock span, temperature/throttle flags and group variance expose failures; (7) keep the 15% limit, add a versioned polling interval; (8) requeue unstable attempts and compare repeated real timings, never rewrite historical verdicts; (9) operator proceeds under user authorisation, with clock locking and whole-kernel cycle rescaling considered and declined for this host. Unit regressions cover stable low clocks, an interior dip, heat and throttling; real noise validation follows from a clean commit. Wall-time remains primary.
