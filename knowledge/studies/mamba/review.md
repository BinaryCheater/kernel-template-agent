# mamba reusable design menu

| Kernel need | Candidate | Useful choice |
|---|---|---|
| Input-dependent affine state | [ordered_affine_prefix@1](delta/ordered_affine_prefix@1/template.yaml) | Ordered composition, identity tails and chunk carry |
| Previous/current innovation update | [shifted_innovation_recurrence@1](delta/shifted_innovation_recurrence@1/template.yaml) | Shift coefficients with matching diagonal/boundary correction |
| Time × rank GEMM | [time_rank_causal_contraction@1](delta/time_rank_causal_contraction@1/template.yaml) | Time quotient mask and full same-time rank block |
| Periodic rotation state | [periodic_phase_prefix@1](delta/periodic_phase_prefix@1/template.yaml) | Chunk prefix on phase quotient |
| SSD tails and packed segments | [chunked_decayed_outer_product_state@2](delta/chunked_decayed_outer_product_state@2/template.yaml) | Identity padding, segment reset and nonpositive decay scope |

Existing rules are bound by actual fields in [the concrete design ledger](designs.md), including scoped exceptions where a seed is too narrow. Read that ledger for one composition, source locations, workload/backend constraints and variant differences. [Focused evidence](checks/designs.md) records executed checks. [Pinned sources](inputs/sources.json) and [pool versions](inputs/pool.json) are retained.

Status: source extraction complete after the depth continuation; proposals await independent review/admission. All materially distinct source groups have representative mechanisms and dispositions. Open native memory-order, backend-layout, floating-point and numeric-domain obligations remain explicit in the rules and ledger; no GPU performance or native validation is claimed. Historical first-pass records remain addressable with the `-initial` suffix.
