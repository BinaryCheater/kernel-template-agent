# Focused structural evidence

Date: 2026-11-04.

Command: `.venv/bin/python knowledge/studies/causal-conv1d/checks/replay.py`

```text
PASS: widths2/3/4, H=W-1..7, multiple starting positions and repeated wraparound
```

Command: `.venv/bin/python knowledge/studies/causal-conv1d/checks/designs.py`

```text
PASS: suffix-history pullback plus stencil adjoint, widths2/3/4 and L1..6, including histories longer than invocation
```

All local candidate Template/ProofOutline pairs validate against current KTA schemas and match name/version. Sources and archives reopen successfully. These checks discharge bounded mapping/algebra witnesses; native GPU lowering, memory visibility, numerical error and performance are not tested here. Each candidate names those remaining obligations.
