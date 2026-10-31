# SYSDEBUG-1 Phase 1 — initial debug handoff

Historical checkpoint before D649. Later clock-repair validation is recorded in SYSDEBUG1.md; the old rejection records remain unchanged.

Date: 2026-10-31 (Asia/Singapore). **Phase 1 exit: BLOCKED.** Visual inspection of real diagnostics is available; the runbook's accepted Phase 2 S0/S1 transition has not occurred. The runbook requires a stop when X6 or X10 cannot be produced.

## At a glance

| Card | Outcome | Effect |
| --- | --- | --- |
| [C1-X6](cards/C1-X6.md) | Blocked: no completed noise estimate; active arm pending | No band applied |
| [C1-X10](cards/C1-X10.md) | Blocked: timed peak/interference jobs rejected | No calibrated HwModel |
| [C1-QUIET](cards/C1-QUIET.md) | Clock-policy question exposed; later interference also observed | Existing policy retained |
| [C1-MEM](cards/C1-MEM.md) | Pending peak RSS and headroom definition | Defaults retained |
| [C1-X11](cards/C1-X11.md) | 3 × 70 model points; rho 0.519–0.536; zero bound violations | Existing thresholds retained |
| [C1-NCU](cards/C1-NCU.md) | Real profiler access works; DRAM metric n/a | No host permission changes |
| [C1-CHECKS](cards/C1-CHECKS.md) | Correct passes; all 3 planted faults caught | Docker invocation repaired |

## Setup and self-tests

- All services run on Spark HK (`KTA_MODE=spark-hk`), store `~/kta/data/dev/SYSDEBUG1`; old `dev/WP23` remains on disk.
- Live frontend: [Pipeline](http://127.0.0.1:14173/pipeline), [Settings → System](http://127.0.0.1:14173/settings?tab=system), [Release](http://127.0.0.1:14173/release). API tunnel: `http://127.0.0.1:17310`.
- Latest source: `e7318d54a2b6ef9bd9510357a5b29df74568d517`, branch `sysdebug/1`; no push or merge. Earlier records retain their own source commits.
- Worker registration includes `gpu=sm_121`, `backend.cuda` and `backend.tilelang`; doctor passes. CUDA and TileLang vector adds each passed exact output checking for 1,048,576 elements.
- Backend lease deferral, timeout recovery and aarch64 CPU compilation passed.
- `kta prove all` ran 7 outlines: shared_memory_tiling reached bounded assurance; the other 6 have no proof level. Recorded counterexamples/unresolved obligations remain as WP24 expected; command exit 0 does not mean every Template proved.
- Proof canary `3e77e0b8…` was caught with replayed counterexamples.
- Codex direct TLS preflight passed, Report `8ab9f304…`; long-run proxy-off policy written through the API with HumanAction `9a412d96…`. Credentials were neither copied nor linked. AgentRun count remains zero; release stage remains S0.
- No accepted Measurement, calibrated HwModel, case-loop GapDecomposition or autonomous agent run is claimed.

## Repairs

1. GPU containers no longer force a Docker runtime named `nvidia`; Spark HK's normal GPU request succeeds. Real checker rerun validates the repair (D645).
2. Measurement CLI now reads the stored controls snapshot, matching the orchestrator; regression distinguishes it from a stale context (D646).
3. Workflow source links now handle `~/kta/src/<commit>/src/...`; the existing source-location assertion is preserved and a deployment-path regression was added (D647).
4. System labels distinguish an optional proxy endpoint from providers set to off, and a missing PATH probe from an absent configured CLI (D648).
5. The remote launcher was pointed at the isolated environment and moved to tmux sessions; previous launcher saved as `~/kta/longrun-WP23.sh`. No old store was overwritten.

## Next supervised session

Start with [the question walkthrough](walkthrough.md) and [open questions](questions.md). First decide how to test quiet clocks, then rerun X6 and X10, review the five cards, and only then begin the accepted S0 chain. S1 calls still require the user's individual release.

Backup and final regression results are recorded in [the delivery note](../SYSDEBUG1.md). All seven cards were independently reviewed against the read-only API. The reviewer agrees with the blocked/pending boundaries; no user acceptance is recorded.
