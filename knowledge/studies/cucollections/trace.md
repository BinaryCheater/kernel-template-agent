# Insertion trace

| Source | Choice / binding | Disposition |
|---|---|---|
| detail/open_addressing/open_addressing_ref_impl.cuh:428-495 | group_size=4, slot_bytes=8, unique keys | cooperative_probe_atomic_claim@1; ballot elects lowest available lane, shfl broadcasts status |
| same file:1700-1738 | relaxed whole packed CAS | Key and payload change atomically; does not order external payload buffers |
| same file:1846-1874 | sizeof<=8 -> packed; sm90 packable wide -> packed; otherwise payload-specific protocols | Narrow candidate deliberately excludes wide paths rather than pretending one CAS covers them |
| probing_scheme.cuh:29,117 | linear vs double hashing | Independent choice; equal keys must have consistent sequences and double-hash stride must cover capacity |
| detail/bloom_filter/bloom_filter_impl.cuh:571-600 | OR atomic scope block/device/system | Scope must include all writers; Bloom false positives are semantic, lost writes would introduce false negatives |
| detail/hyperloglog/hyperloglog_impl.cuh:250-320 | shared accumulation if capacity reservation succeeds, global otherwise | Private sketch must merge with same precision/hash; occupancy policy is not correctness proof |

Worked collision: A and B see the same empty slot. A CAS replaces it; B receives A and retries or reports duplicate if equal. Advancing past the full group after a failed CAS could skip another available slot and break same-key consistency. Source explicitly continues the round on interference.

Publication, full-cycle probe and concurrent device memory semantics are obligations, not checked by a sequential collision example. No executable or GPU test performed.
