# Cascaded and split attention states

When KV is segmented, preserve a state representation and the exact query/support
identity at each merge. Current include/flashinfer/attention/cascade.cuh and
flashinfer/cascade.py in the [retained tree](inputs/sources.json) provide the
following concrete uses of existing rules.

| Source choice | Rule and binding / gap | Consequential contract |
|---|---|---|
| cascade.cuh:374-475 persistent variable-length merge | split_combine_kernel@1 rescaled_merge,merge_grid persistent | Static CTA stride over request/head outputs; o_indptr defines exactly the contributing normalized output/base2-LSE states. Zero states write zero/-inf; one copies. Inactive graph slots do not participate. |
| cascade.cuh:125-145,450-464 | smem_partial_combine@1 serial,rescaled_merge,all_threads | Each y owner first normalizes its local numerator and stores (O,LSE). The helper restarts a raw state and merges each pair as raw(o=O,m=LSE,d=1), then normalizes again. The two normalizations surround a representation change; deleting the second or dividing by the old local denominator again is wrong. |
| cascade.py:401-516,550-569 multi-level wrapper | Complete support partition and normalized-state merge composition | Each level computes outputs for the same Q against its own KV segment, then merges states in place. Only the last level uses the requested causal mode; earlier levels are treated as prefixes. Segment overlaps require explicit multiplicity semantics; a merge does not deduplicate KV. |
| cascade.py:787-825 shared-prefix decode wrapper; :829 onward prefill relative | Query batching/output tiling plus state composition | Shared prefix uses noncausal single-prefill on the batch of Q rows; unique suffix uses per-request paged attention. Merge states for each original Q/head. It shares KV transport across queries, not a query-independent attention output: different Q rows still need their own score/statistic state. |
| merge_state/merge_state_in_place public operations | Existing rescalable-merge algebra; alias/support binding | Inputs must share query identity, exponent/log base, score policy and disjoint logical support or explicit intended multiplicity. In-place mode must retain old input state until every reading lane finishes. These premises do not follow merely from equal tensor shapes. |

For two segments with masses2 and6 and normalized scalar outputs3 and5,
the merged output is(2*3+6*5)/8=4.5, not the unweighted mean4. Treating a
normalized segment output as its raw weighted numerator gives a different
answer. A common key appearing in both segments contributes twice unless a
separate support construction removes that overlap.

The state algebra and global split-combine are already available in the pool;
no new candidate is needed here. CUDA merge-state empty sentinels, data-stage
barriers, per-thread shared readers and PDL waits have concrete source bindings
but are not claimed as native correctness evidence. Combined POD execution
shares the state algebra while introducing a distinct schedule and remains
open in the library inventory.
