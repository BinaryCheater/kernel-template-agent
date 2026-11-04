# causal-conv1d reusable design menu

| Kernel need | Candidate | Useful choice |
|---|---|---|
| Streaming depthwise stencil | [causal_window_state@1](delta/causal_window_state@1/template.yaml) | Ring/linear history and caller-owned position |
| Differentiable final history | [history_suffix_pullback@1](delta/history_suffix_pullback@1/template.yaml) | Route suffix cotangents to input and initial history |

Existing rules are bound by actual fields in [the concrete design ledger](designs.md), including scoped exceptions where a seed is too narrow. Read that ledger for one composition, source locations, workload/backend constraints and variant differences. [Focused evidence](checks/designs.md) records executed checks. [Pinned sources](inputs/sources.json) and [pool versions](inputs/pool.json) are retained.

Status: source extraction complete after the depth continuation; proposals await independent review/admission. All materially distinct source groups have representative mechanisms and dispositions. Open native memory-order, backend-layout, floating-point and numeric-domain obligations remain explicit in the rules and ledger; no GPU performance or native validation is claimed. Historical first-pass records remain addressable with the `-initial` suffix.
