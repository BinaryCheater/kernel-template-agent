# Recorded local check

Command: `.venv/bin/python knowledge/studies/causal-conv1d/checks/replay.py`

Date: 2026-11-04.

```text
PASS: widths2/3/4, H=W-1..7, multiple starting positions and repeated wraparound
```

Schema validation: Template and ProofOutline accepted by current kta.core.schemas; identity matches. Retained source archive, full file listing and license files reopen successfully.

This is a bounded structural replay of the source-derived mapping, not GPU execution, native lowering verification, or a performance measurement. Floating-point error, synchronization and backend layout obligations remain open as named in the trace.
