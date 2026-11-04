# Source bindings and decisions

All paths relative to the pinned tree in inputs/sources.json.

| Source / binding | Rule field or gap | Evidence |
|---|---|---|
| indices.py:294-337, one input per loop and blockIdx.y offset | offset_indexed_sparse_incidence@1 program | Integer query predicate precedes atomic pair reservation; transposed switches query |
| indices.py:450-500, bounded output set | capacity premise | Dropped output identities require recompacting pair counts; cannot reuse old counts |
| indices.py:600-650, offset=33 | mask_word_bits=32 | word=1, bit=1; atomicOr merges independent offsets while pair[offset,output] has one writer under unique-coordinate premise |
| indices.py:748-805, submanifold | output domain binding | Center identity map; other offsets pair reversed with RS-1-offset; only matching sites survive lookup |
| ops.py:811-1097 native, 1450-1666 implicit | output_tiling@2 conditional reuse; reduction_partition@1 conditional reuse | Native materializes gather and scatter; implicit consumes incidence/mask directly. GEMM arithmetic implementation delegated to cumm |
| pointops.py:232-246 | Gap: bounded append | Atomic reservation occurs before old<capacity test; count may exceed stored points until clamp, so do not treat count as initialized payload extent |
| maxpool.py:211-255 | reduction_partition@1 conditional on assoc/comm and nonempty count | Output average divides by present neighbor count; absent entries skip and zero count returns zero |

Composition: geometry -> deduplicated output identities -> offset pairs/masks -> channel contraction -> per-output accumulation. Mask sorting is a schedule and cannot change the incidence relation. No performance threshold inferred from dispatch.
