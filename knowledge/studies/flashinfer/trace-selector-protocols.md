# DSA selection relatives: publication and reuse protocols

Sources are in `flashinfer/experimental/sm120_dsa_indexer/csrc/` under the
retained [revision](inputs/sources.json). This trace binds shared pipeline
concepts and names protocol differences after the
[exact-selection extraction](trace-exact-indexer-selection.md).

| Source mechanism | Existing rule fields or gap | Concrete source disposition |
|---|---|---|
| `sm120_dsa_indexer.cuh:2506-2544`: math group completes score block and range metadata, then signals res_ready[iteration % slots] | per_slot_handshake@1 generation/readiness, multi_buffer_rotation@2 | Published payload is score rows plus row identity and ranges. Readiness follows all math writes; row-range reduction scratch is separately protected before its next use. It is not a readiness event per individual score. |
| Same kernel resident selector role consumes completed rows while math produces later blocks | Generic role specialization plus resident-slot realization gap | This is score production/selection overlap, not a new score approximation. Exact selected set and original payload mapping retain radix_boundary_topk@1's order contract. Slot claim/last-reader retirement still need the selected native helper binding; the arrival above alone does not prove safe overwrite. |
| `decode_select_long.cuh:60-86`: disjoint contiguous row slices, sense-reversing row barrier with count and generation | Shared completion concept, not epoch_tagged_completion@1 field match | Leader fences, performs acquire-release count increment, last resets count and release-increments generation; others acquire-wait for that generation to change. The old rule fixes a 64-bit 128-wide epoch/count encoding and <=64 participants; it cannot be relabeled as this separate 32-bit count/generation barrier. |
| `decode_select_long.cuh:343-372,422-430,479-482`: invocation epoch chooses parity of histograms, lists/output counters; initialize next parity while current data is in use | physical_storage@1 lifetime; parity-state realization gap | All row CTAs observe epoch before arrival. An arrival-selected CTA advances invocation epoch after its own emit, which is not proof that all other CTAs have completed. Serial nonoverlapping calls and the later grid completion protect cross-call reuse; epoch increment alone grants no overwrite permission. |
| `decode_select_long.cuh:158-257,443-497`: coarse cut, direct emit above cut, list and exact ranking when it fits, otherwise refine cut range through 4096 bins and finally ID | Same rank-boundary invariant; coarse-region refinement gap | This is not one of radix_boundary_topk@1's fixed 32-bit digit schedules. All emitted-better terms decrement the unresolved rank; next histogram counts only the unresolved cut region. Exact score payloads remain separate from coarse bucket indices. A one-value region still needs original-ID tie selection. |
| `decode_select_long.cuh:500-508`: row CTAs wait on one another; no early dependent trigger | Progress/residency binding | Source intends long-row waiters to fit one resident wave while short rows never wait. An early dependent grid could consume slots awaited by unscheduled row CTAs. The source comment is a required launch premise, not measured residency evidence. |

The useful composition is complete score publication -> exact boundary
membership -> caller-specific output presentation. Row barriers and parity
state protect generations but do not alter ranking. No new numeric candidate
is needed for each resident length class. The unresolved coarse-region rule,
resident slot claim/retirement realization and reusable sense-barrier protocol
are named gaps; they do not invalidate the already extracted finite ordering
and full-count certificates.

Keep ragged IDs ascending and paged IDs as an unordered selected set with
aligned scores and tail padding. Respect the caller's shape bound and
nonoverlapping per-device paged scratch contract. Native proxy/fence semantics,
last-arrival progress and graph replay remain evidence obligations; no device
execution or timing was performed for this trace.
