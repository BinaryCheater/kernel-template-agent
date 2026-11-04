# cuCollections reusable choices

Status: extracted by distinct NVIDIA design; pending independent review.

| Kernel need | Rule | Choice |
|---|---|---|
| Claim colliding hash slots | [Packed claim@2](delta/cooperative_probe_atomic_claim@2/template.yaml) | Cooperative group and whole-slot atomic width; old@1 retained |
| Key/value wider than atomic slot | [Split publication](delta/split_slot_claim_publication@1/template.yaml) | Stable key/payload readiness versus insert-only speculation |
| Remove entries without breaking lookup | [Tombstones](delta/tombstone_preserving_probe_chain@1/template.yaml) | Preserve the probe chain through deleted slots |
| Grow without migrating old data | [Segmented maps](delta/segmented_submap_growth@1/template.yaml) | Designated target and search across existing segments |
| Retrieve many matches | [Output buffering](delta/coalesced_match_output_buffer@1/template.yaml) | Shared paired records and global reservations per flush |
| Compact approximate membership | [Blocked Bloom](delta/blocked_hash_membership_filter@1/template.yaml) | Block width, atomic word OR and shared pattern policy |
| Fixed-memory cardinality state | [Max-rank sketch](delta/hashed_max_rank_sketch@1/template.yaml), [private max](delta/privatized_max_summary@1/template.yaml) | Precision/hash mapping versus local placement/merge |
| Exact density-adaptive membership | [Key containers](delta/adaptive_key_partition_membership@1/template.yaml) | Sparse arrays, dense bitsets and imported run intervals |

Composition: a growing map targets segment1 while checking segment0 first. Existing key7 cannot be inserted again merely because the target is empty. New wide-slot values follow the selected packed or split publication contract. A multimap retrieval buffers paired matches; bucket2 gives maximum step64 and buffer96, so pending40 flushes before another all-hit step.

[Source evidence and variant bindings](trace-additional-designs.md) cover every inventory row. [Local replay](checks/replay-result.json) distinguishes full-cycle probing, tombstone reachability, bounded append/flush, max versus additive merge and inclusive run endpoints. Command: `.venv/bin/python knowledge/studies/cucollections/checks/replay.py`. Ten Template/ProofOutline pairs pass current schemas, including both packed-claim versions.

The paired outlines state invariants and pending device obligations; local algebra is not a concurrent memory-order proof. In particular speculative payload insertion excludes readers until quiescence, and HLL state equivalence does not certify estimator error. [Migration](delta/migration.md) records the16-byte domain extension without overwriting the original version. No GPU execution or performance measurement.
