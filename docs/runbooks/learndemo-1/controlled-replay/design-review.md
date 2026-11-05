# Independent design review

Outcome: no remaining material contradictions after corrections. The protocol is ready for preparation, not for paid implementation. Source build, oracle/evaluator, actual packages, cache/access/budget enforcement and measurement receipts remain unverified. All prelaunch flags remain false.

The reviewer was given the user's research question and instructed to attack the source-to-Template-to-reconstruction comparison, rather than repeat the old campaign's internal artifact audit. No model implementation or GPU experiment was invoked for this review.

Verified against retained source: exact FA2 revision, translation unit, launcher, GB10 branch, BSHD O layout and natural-log LSE. Verified exposure inventory: ten canonical plus eight explicitly unpaired proposals, 18 prescribed and 320 full-library entries. The treatment is mapped pooled knowledge, not an extraction solely from that revision.

| Defect found | Correction in the protocol |
|---|---|
| Eager correctness could pass while timed graph replays stale results | Test both paths on changed values at captured addresses; verify O/LSE and mutation before/after replay; add stale-output canary |
| Unchanged original source could still be compiled into a weak comparator | Require upstream-equivalent release optimization flags and independently review compatibility changes |
| Machine oracle formulas omitted the causal mask | Explicit raw scores -> masked scores -> softmax/O/LSE formulas |
| Public causal/window/dropout arguments differ from internal source fields | Record upstream right-window normalization, stored causal window and keep-probability conversion |
| Discovery, final confirmation, retries and queue-time enforcement were ambiguous | Fix model/confirmation caps, retry scope, live queue-adjusted watchdog and worker timeout; require enforcement checks |
| Ratio and drift estimators could change borderline verdicts | Specify ratio of median block times, inclusive quartiles and max/min−1 drift formulas |
| Pooled provenance and shared compiler artifacts could overstate information control | State pooled-library claim; freeze document allowlist and verified empty per-slot caches |

The coordinator also fixed an axis mismatch risk: new inputs are BSHD, so cancellation and future-value patterns must use sequence axis1 rather than reusing the old BHSD generator unchanged.

This review accepts the design for preparation only. It does not supply any missing runtime receipt or certify the original kernel runnable. P7 requires another question-to-evidence review of P1-P6 execution artifacts before sealing the campaign.
