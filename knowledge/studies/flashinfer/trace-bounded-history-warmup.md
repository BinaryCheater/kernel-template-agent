# Independent recurrent intervals after a bounded warm-up

When a long state chain loses its boundary dependence fast enough, independent
items can replay a short overlap from zero before publishing their own output
interval. [bounded_history_warmup@1](delta/bounded_history_warmup@1/template.yaml)
and [outline](delta/bounded_history_warmup@1/proof.yaml) capture this approximate
history choice, its error budget and the ownership fallback when a cut fails.

The concrete source is the SM90 merged GDP prefill split, in the retained
[tree](inputs/sources.json). Its scope differs from exact expanded-state chunks:
chunked_delta_feedback@2 preserves the incoming state; this design discards it
at admitted warm-up starts. grouped_state_substeps@1 supplies real-token versus
expanded-update coordinates.

| Source decision | Rule field / concrete binding | Evidence |
|---|---|---|
| `cake_gdp_backend/cake_jit.py:3890-4000`:64-update chunk pairs, up to48 candidate items, alignment lcm(2,n), source span/cost policy | `planned_cut_count<=48`, `chunk_updates64`, `pre_partition` | Spacing trades parallel waves against repeated warm-up/seed/pipeline costs. Cost constants and minimum span by head count are tuning choices, not correctness premises |
| Concrete `cake_gdp_245414df8449fb1825a1_kernel.cu:5550-5914`: read(seq,head,cut,span); sum log2(gate+1e-10) over first substeps only; inspect up to8 preceding chunks; stop at cumulative<=-24 | `max_gate_scan_chunks8`, `log2_gate_threshold-24` | cut0 is always active. A nonzero unsaturated cut has no work. The source numerical gate test is not a certificate for arbitrary raw K/V/initial state |
| `:5915-6296`: forward scan at most48 later planned cuts, stop at next saturated cut or end; compute_start=cut-scan_w, back up one extra chunk to make the executed interval even | `pre_probe`, `pre_ownership` | A rejected cut extends its previous active owner's interval. Up to8 scanned warm-up chunks can become9 executed chunks after pair alignment; use the actual start in the seed bound |
| `:1849-1850,1922`: only cut0 uses the original seed; later active items initialize their local state to zero | `cut_policy=bounded_reset`, `pre_state` | Overlap recomputation approximates omitted history; it is not a cached exact checkpoint |
| `:2869,3642,4417,5190`: chunk_writes predicates begin at write_lo=cut | Warm-up is read/recompute only | Overlapping warm-up regions never duplicate output writes. Only interval ending at global end owns final state |
| `:2102-2110`: first original-seed reader group completes, threadfence then publishes a per-sequence/head flag; `:5207-5224`: nonfirst final owner acquire-waits and joins state-store participants; `:5368`: clear flag | `pre_lifetime` | Original seed read completion precedes final-state write. The numerical argument does not prove flag epoch, GPU progress or alias contracts |
| `cake_jit.py:4043-4094`: per-sequence body cost route; stable body classes launch disjoint gathered sequence sets | Ownership reuse of explicit work classes | A sequence's choice depends on its own chunk count/head/substeps/device, while unit/seed/final-state flags restrict menus. These are dispatch/runtime preparation details, not additional state algebra |

## What the gate test does and does not establish

For V-major delta state with identical inputs, difference evolves as
Delta_next=alpha*Delta*(I-beta*k*k^T).
In spectral2 norm, the correction factor has norm
max(1,abs(1-beta*||k||^2)) for key dimension>1. Thus0<=beta*||k||^2<=2
makes each correction nonexpansive, and scalar gate product rho bounds the
remaining state difference by rho*M, where M bounds the exact state at the
actual zero-reset start. Scaled readout error is at most||scale*q||*rho*M.
Rounding/approximation perturbations require their own additional budget.

The candidate requires these bounds explicitly. Source prefill accepts raw
K/V and initial state; its cut admission tests recent gate decay, not key
contraction, omitted-state magnitude or readout magnitude. Therefore the
source threshold is a useful heuristic binding on an admitted input/error
domain, not proof of a universal tolerance. Even normalized keys do not bound
absolute error without an omitted-state/value bound. In the first candidate,
`pre_transport`, `pre_state` and `pre_budget` expose this missing applicability
information instead of silently asserting it.

The [exact check](checks/check_bounded_history_warmup.json) verifies180 prefix
state bounds and360 readout bounds using independent affine and correction
recurrences under nonexpansive keys/beta. Two consequential counterexamples:

- gate2^-24 times an omitted state2^30 still leaves absolute state error64;
- gate2^-24 with beta1 and key norm2^13 amplifies unit-state error by almost4.

The same check enumerates255 active-cut patterns,11778 owned chunk occurrences
and342 extra-leading alignment cases. It compares forward-scan ownership with
an independent active-cut interval list: complete output coverage, warm-up
suppression and exactly one final-state owner. It does not validate native
log/exp upper bounds, rounding budgets or the seed-reader GPU protocol.

Composition: grouped_state_substeps@1 preserves real-token write boundaries;
chunked_delta_feedback@2 supplies each item's exact residual/state organization;
bounded_history_warmup@1 permits replacing its incoming state only under the
stated error budget, while exact_only retains the original chain. No existing
retained materialization/checkpoint rule expresses a bounded omitted history
and rejected-cut interval extension, so this is a distinct proposal.

This disposes the merged gate-saturation split mechanism and its explicit
numerical applicability gap. Other recurrent scheduling, locality, state
propagation, codecs and training families still need their own concrete
representatives in the [inventory](recurrent-inventory.md).
