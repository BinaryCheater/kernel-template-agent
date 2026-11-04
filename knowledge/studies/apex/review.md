# apex reusable design menu

| Kernel need | Candidate | Useful choice |
|---|---|---|
| Many small tensors | [bounded_chunk_launch_table@1](delta/bounded_chunk_launch_table@1/template.yaml) | Chunk table capacities, flushes and global tensor identity |
| North/west dependent DP | [diagonal_dependency_wavefront@1](delta/diagonal_dependency_wavefront@1/template.yaml) | Anti-diagonal frontier and independent edge prefetch |
| Small peer halo messages | [alternating_peer_flit@1](delta/alternating_peer_flit@1/template.yaml) | Two slots with consumed phase tracking |
| Peer statistic replication | [tagged_peer_reduction_rounds@1](delta/tagged_peer_reduction_rounds@1/template.yaml) | Disjoint doubling rounds and epoch-tagged payloads |
| Local permutation search | [local_change_score_delta@1](delta/local_change_score_delta@1/template.yaml) | Cancel unchanged objective groups |
| Vectors cross normalization groups | [group_aligned_vector_partials@1](delta/group_aligned_vector_partials@1/template.yaml) | Use gcd-width microblocks before group combine |
| Tiny normalized gradient checkpoints | [deferred_gradient_scalar@1](delta/deferred_gradient_scalar@1/template.yaml) | Delay common scalar until after reload |

Existing rules are bound by actual fields in [the concrete design ledger](designs.md), including scoped exceptions where a seed is too narrow. Read that ledger for one composition, source locations, workload/backend constraints and variant differences. [Focused evidence](checks/designs.md) records executed checks. [Pinned sources](inputs/sources.json) and [pool versions](inputs/pool.json) are retained.

Status: source extraction complete after the depth continuation; proposals await independent review/admission. All materially distinct source groups have representative mechanisms and dispositions. Open native memory-order, backend-layout, floating-point and numeric-domain obligations remain explicit in the rules and ledger; no GPU performance or native validation is claimed. Historical first-pass records remain addressable with the `-initial` suffix.
