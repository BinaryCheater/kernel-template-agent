# cuCollections NVIDIA design inventory

Status: extracted by distinct design. Public include/cuco container headers were reconciled with detail implementations and bulk kernel adapters. Paths below are relative to include/cuco/; [concrete decisions](trace-additional-designs.md) provide source lines and bindings.

| Design family | Representative and differences | Disposition |
|---|---|---|
| Cooperative probing | Linear/full-cycle double hash, group lanes and bucket widths | Packed claim1/2 with explicit probe-cover premise |
| Atomic slot width | Packed4/8 vs native packable16 at sm90+ | Version2 widening; [field migration](delta/migration.md) preserves version1 |
| Split wide payload | Key-dependent stable write/wait vs insert-only speculative two-CAS cleanup | Split publication candidate; mixed-reader/erase restrictions explicit |
| Unique/multi containers | Map/set duplicate rejection vs multimap/multiset multiplicity | Shared device body with explicit semantic mode; no unique-key claim applied unchanged to multi insertion |
| Erase/find/update | Tombstone and empty semantics, stable observed-slot transitions, found-payload updates | Tombstone candidate; arbitrary mixed-operation linearizability remains a scope premise |
| Multi-match retrieval | Ballot offsets, shared record batches, atomic global flush reservations | Coalesced paired-output buffer candidate |
| Dynamic growth | Geometric fixed-submap append, cross-submap duplicate search, designated target | Segmented growth candidate; no migration or concurrent independent targets assumed |
| Blocked Bloom | Split hash, pattern OR, cooperative word tests, OR merge/AND intersection | Membership filter candidate; approximate positive answers and monotone insertions separated |
| HyperLogLog | Hash-prefix register, suffix CLZ rank, max merge, HLL/HLL++ finalization | Hashed max-rank candidate; estimator accuracy separate from state identity |
| HLL placement | Direct global vs shared-local max array, vector load and single tail owner | Private max-summary candidate; distinct from CCCL additive histogram rule |
| Roaring build | Sortedness menu, sort/unique, prefix payload offsets, array/bitset work queues | Adaptive container candidate; one owner per completed bitset word |
| Roaring query/import | Prefix lookup, arrays/bitsets, imported runs,32/64-bit hierarchy, misalignment | Same exact set semantics with explicit serialized format and inclusive endpoints |

All major public container mechanisms have concrete dispositions. Remaining realization obligations: packed128 lowering, split-slot mixed-operation reader safety, native atomic scopes, valid serialized offsets, and statistical accuracy models. No major unread family is used as a closure placeholder.
