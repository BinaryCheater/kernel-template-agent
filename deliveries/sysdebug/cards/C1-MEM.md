# C1-MEM Can the unified-memory CPU cap be calibrated?

Level: A · Status: reviewed

**Data:** The idle timing record reports about 7.48 GiB host memory used, but compile/check peak RSS, a measured GPU peak and a headroom margin are not available. [Record](http://127.0.0.1:14173/pipeline?rec=acd658ebd5791db166ccab9408777febedc09239dd51d7a790ce797402fd5dcd).

**Rule:** cap = total − measurement peak − stated margin.

**Result:** PENDING. Leave the existing default labelled default; no new cap selected.

**If wrong:** A cap inferred from idle memory could starve the GPU or kill concurrent jobs.

**Reviewer:** Independent Codex reviewer, separate session (2026-10-31): Agrees with PENDING — idle used memory cannot establish a safe concurrent-job cap.

**Decision:** No user decision requested or recorded; no research control value changed.
