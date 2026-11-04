# Decisions and bindings

| Source | Binding / category | Rule or gap |
|---|---|---|
| search_plan.cuh:105-137 (under cpp/src/neighbors/detail/cagra/) | AUTO single CTA if itopk<=512 and max_queries>=2*num_SM; persistent forces supported single CTA path | Dispatch heuristic, not legality or speed evidence |
| search_multi_cta.cuh:119-127 | local_topk=32; num_cta=max(width,ceil(global_itopk/32)); internal 96,width2 -> 3 | split_scope_traversal_deduplication@1 instance |
| search_multi_cta.cuh:181-202 | intermediate count=3*32=96, must cover requested topk | Derived storage and legal output capacity |
| search_plan.cuh:265-310 | local visited bits 8..14; global traversed bits 11..25, bounded growth | Local vs global lifetime; fill-rate sizing is policy subject to capacity premise |
| search_multi_cta_helpers.cuh:16-80 | ranked candidate node17, two CTA claimants | Global insert winner marks MSB, loser invalidates local copy; unique expansion is distinct from unique final ID filtering |
| search_multi_cta_kernel_launcher_jit.cuh:86-90 | initialize query traversed tables before launch | Publication edge; no source-level claim of kernel execution |

Composition: planner sizes per-query tables -> per-CTA candidate exploration -> shared parent claim -> per-CTA results -> final top-k. Ownership claim is valid independently of nearest-neighbor recall; finite exploration can miss a closer unseen point.

Source substitution checked num_cta=3 and capacity96; atomics and marker-bit address recovery remain realization obligations. No custom executable replay or GPU measurement performed.
