# C1-NCU Can the operator profile on Spark HK?

Level: A · Status: reviewed

**Data:** Profile 39fb0f72242945e6588af2a26a95eb09973b8b082974d944afe107dff5445c3b from the real KTA Executor: gpu__time_duration.sum = 1,176,224 ns (1,176.224 µs); SM throughput = 83.19%. Raw CSV reports requested DRAM throughput as n/a. Earlier independent profile also succeeded. Driver RmProfilingAdminOnly = 0. [Profile](http://127.0.0.1:14173/pipeline?rec=39fb0f72242945e6588af2a26a95eb09973b8b082974d944afe107dff5445c3b).

**Rule:** Demonstrate counters through the normal worker user and Executor; distinguish permission from individual metric availability.

**Result:** Profiling access VERIFIED. No sudo, driver change, daemon change, capability grant or reboot needed. DRAM metric coverage remains a gap.

**If wrong:** Treating partial metrics as full coverage would hide a profiler capability gap; treating raw ns as µs would introduce 1000× error.

**Reviewer:** Independent Codex reviewer, separate session (2026-10-31): Agrees — profiler access works; duration units and unavailable DRAM coverage are reported accurately.

**Decision:** No user decision requested or recorded; no research control value changed.
