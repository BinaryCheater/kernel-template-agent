# Representative bindings

| Source | Concrete choice | Rule / evidence |
|---|---|---|
| utils/WarpSelectFloat.cu:14-80 | k=33 -> warp queue 64, local queue 3 | collective_triggered_candidate_queue@1 capacities; power-of-two menu, CUDA warp 32 |
| utils/Select.cuh:190-237 | any lane full triggers merge | no_overflow: test after each add, collective vote, clear all lanes after merge |
| utils/Select.cuh:288-305 | final flush, shared block merge | collective_complete; CTA barrier separates independent warp queues from cross-warp merge |
| utils/Select.cuh:450-540 | warp retained pairs in registers | Same queue invariant, different physical storage |
| utils/Comparators.cuh:17-65 | default key-only vs key/value comparator | Default equality does not imply smallest-ID winner; sorted_run_topk@1 smallest-ID contract cannot be applied unchanged |
| impl/Distance.cu:290-350 | select on centroid_norm-2dot then add query_norm | Row-constant rank invariance in exact arithmetic; finite rounding and nonnegative clamp limit numerical equivalence |
| impl/PQCodeDistances-inl.cuh:28-100 | one thread per code, block per subquantizer | Reuse code centroid registers over queries, share query/residual vector; full double-buffer schedule not reconstructed |

Composition: query/database tile -> dot products and centroid norm -> local top-k queue -> tile winners -> final selection. Local top-k pruning is safe only with a compatible order at every merge. Source substitution distinguishes local queue=3 from retained queue=64; they are independent capacities subject to compiled menu and resources.
