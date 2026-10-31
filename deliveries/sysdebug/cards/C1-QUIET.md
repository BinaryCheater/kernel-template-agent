# C1-QUIET Does the quiet-host rule distinguish interference from normal clocks?

Level: A · Status: reviewed

**Data:** 16 rejected attempts retained as Reports; 0 Measurements. The first four otherwise-quiet vadd attempts were rejected only because 1 − min(start,end SM clock)/NVML maximum was 0.6234, 0.1971, 0.1625, 0.1625 > 0.15. CPU 0.10–0.20 cores, temperature 47–49 C, throttle flags 0. Later attempts also include genuine CPU/GPU activity and must not be pooled as idle. [Record](http://127.0.0.1:14173/pipeline?rec=acd658ebd5791db166ccab9408777febedc09239dd51d7a790ce797402fd5dcd).

**Rule:** Thresholds = an idle distribution plus an explicit margin; a deliberate overlapping compile must be flagged. No margin or alternative clock statistic was fixed by the runbook.

**Result:** BLOCKED policy question. The implementation measures distance below rated maximum, not a change in clock during the kernel. No threshold was changed; no failed attempt was converted to a Measurement.

**If wrong:** An unsuitable clock test can reject ordinary DVFS; removing it without replacement can accept throttled timings.

**Reviewer:** Independent Codex reviewer, separate session (2026-10-31): Agrees with BLOCKED — clock-only idle rejects are correctly separated from later activity-related rejects.

**Decision:** No user decision requested or recorded; no research control value changed.
