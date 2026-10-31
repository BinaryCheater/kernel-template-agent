# C1-CHECKS Do the real correctness checkers catch the planted faults?

Level: A · Status: reviewed

**Data:** Report c907edb357877d2497ca01f7cfa23eef3959b58fa910ecf82323394837f1b530: correct GEMM passes; oob_write fails memcheck; missing_row fails NaN coverage and numeric checks; cublas fails library and kernel-trace checks. All 4 verdicts match expectations. [Report](http://127.0.0.1:14173/pipeline?rec=c907edb357877d2497ca01f7cfa23eef3959b58fa910ecf82323394837f1b530). Before the fix all variants had a reference-launch error, preserved in f36a51d4….

**Rule:** The correct module must pass and each of the three planted faults must be caught by its expected checker.

**Result:** PASS after removing the forced, unregistered Docker runtime. GPU device requests, worker uid and network isolation are preserved.

**If wrong:** A broken reference path could masquerade as a kernel correctness failure.

**Reviewer:** Independent Codex reviewer, separate session (2026-10-31): Agrees — the four real Check records match all expected verdicts after the container fix.

**Decision:** No user decision requested or recorded; no research control value changed.
