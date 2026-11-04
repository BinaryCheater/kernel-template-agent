# DSA indexer: count-certified filtering and radix boundaries

Selected route: `flashinfer/dsa_indexer.py` with backend `sm120` ->
`experimental/sm120_dsa_indexer/_api.py` -> `csrc/binding.cu` and
`csrc/sm120_dsa_indexer.cuh` in that experimental directory. All paths refer to
the retained [revision](inputs/sources.json). The JIT at `_api.py:43-59` builds
for SM12x without fast math. This trace covers exact selection of supplied
finite FP32 scores; the source's claimed bit identity to DeepGEMM scoring is
not established by these logical checks.

Two new objects are separate:
[count_certified_topk_filter@1](delta/count_certified_topk_filter@1/template.yaml)
controls whether a bounded candidate list is sufficient, while
[radix_boundary_topk@1](delta/radix_boundary_topk@1/template.yaml) implements
exact selection of that list or a complete fallback domain. Each has its own
proof outline.

## Long ragged rows: threshold changes work, not winners

| Source decision / actual binding | Candidate field | Evidence or boundary |
|---|---|---|
| `_api.py:28-29,184-188`: candidate capacity 8192, requested k<=4096; target min(2*k,sqrt(k*capacity)) | `capacity`, boundary `k`; threshold estimator is caller binding | The target is not a certificate. Changing it can change fallback frequency without changing the requested result. |
| `sm120_dsa_indexer.cuh:1068-1130`: sample exact scores, choose a sample rank; no sample or a row fitting capacity gives tau=-inf | Program threshold input / `pre_filter` | Sampling gives no probabilistic correctness premise. The following complete scan decides acceptance. Non-NaN threshold and finite valid scores are the extracted domain. |
| `:2467-2470`: keep all valid scores >=tau; warp-local staging then flush | `pre_filter`, `pre_append` | Every valid key is considered; equality is retained, so a large boundary tie cannot silently disappear. |
| `:2310-2321`: atomicAdd count by the whole flush size, write only reserved slots below capacity | `count_and_storage` | The count includes overflow entries even when their payload is not written. A saturated or stored-record count would destroy the certificate. Atomic slot assignment must remain unique. |
| `:1181-1197`: want=min(k,visible), reject c<want or c>capacity and flag row; discard candidate storage for histogram reuse | `filter_certificate`, `pre_lifetime` | Only the full interval want<=c<=capacity admits candidate selection. Underfull candidates do not justify returning fewer winners. |
| `binding.cu:499-519`: scan/finalize, then exact-mode dispatch reads flagged row list | `fallback_identity`, `pre_score` | Full-domain rescoring must preserve the exact score bits, valid domain and tie comparator, even across kernel variants. Phase order and histogram reuse require real publication evidence. |

If at least want keys pass a complete scalar threshold, every discarded score
is below those qualifiers. Therefore all top-want winners are in the complete
qualifier set. The upper count bound proves that set was fully stored. An
unlucky sample can make the candidate set too small or too large; either case
uses exact fallback. Accepting only the first capacity append records instead
can lose high-score keys that arrive later.

## Exact boundary and equal-score identities

| Source decision / actual binding | Candidate field | Evidence or boundary |
|---|---|---|
| `sm120_dsa_indexer.cuh:154-163`: unsigned monotone FP32 key, negative bits complemented, nonnegative sign bit set, signed zeros folded | `pre_encoding` | Scores rank descending, then larger original key ID. Without zero folding +0 would beat -0 regardless of ID. NaN ranking is outside this candidate. |
| `:998-1065`: four most-significant-first 8-bit passes | `digit_schedule=bytes_8_8_8_8`, `rank_prefix` | Histogram only the current prefix, scan digits descending, subtract strictly better-bin counts. Carry residual rank, not the original k. |
| `:1200-1252`: score-boundary key and residual tie count; if needed select the tied IDs, then keep scores above boundary or equal scores with ID>=cut | `tie_identity`, `pre_emit` | Tie selection reads only original IDs whose score equals the boundary. It does not apply one combined approximate score/ID key. |
| `:397-418,782-879`: exact fallback uses 2048-bin 11/11/10 score passes; same ID passes only when all boundary ties would overflow capacity | `digit_schedule=bits_11_11_10`, distributed generation obligation | Above count plus equal count determines whether tied IDs must be reduced before candidate collection. If all ties fit, retain them and let finalization choose the required IDs; skipping ID passes does not mean all ties are final winners. |
| `:2428-2451`: exact collect predicates and bounded final candidate writes | `pre_emit` | After the cut, at most capacity candidates are retained by construction. The finalizer still enforces exactly want selected identities. |
| `:1253-1345`: present final winners in ascending ID order, using bitmap-prefix ranks when the key domain fits, otherwise sorting selected complemented IDs | Output presentation contract | Winner ranking favors large IDs on score ties; presentation sorts the chosen IDs ascending. Associated score payloads move with IDs. Tail slots are -1/-inf. Bitmap ownership, score lookup and selected-ID sorting are backend bindings. |

`sorted_run_topk@1` is not direct reuse: its k domain is at most 32, its
algorithm merges sorted runs, and its ties favor the smallest identity. Neither
calling it with k=2048 nor silently reversing its tie semantics is a legal
binding. The new radix candidate retains that older version unchanged.

The [logical check](checks/check_exact_selection.json) runs 13,104 threshold
cases over finite values, empty rows, requested k, capacity and tau; it checks
both candidate acceptance and exact fallback against full sorting. Overflow
and underfull witnesses distinguish false certificates. Seventy-two radix
cases exercise both digit schedules, negative scores, signed zeros, altered
input order and all relevant ranks. A crowded tie selects IDs 37..41 from
42 entries with k=5. Without signed-zero folding a smaller +0 ID incorrectly
beats the larger -0 ID. These are abstract ordering/count checks, not native
atomic, fence, warp or score-bit validation.

## Relatives and remaining bindings

Resident ragged rows (`_api.py:159-181`, `binding.cu:350-387`) select among
shape-bound length classes. The source describes math/selector overlap using
reused scratch slots; its data-ready and reader-retirement frontiers still need
a concrete role binding. Longer paged decode uses `decode_scan.cuh` plus
`decode_select_long.cuh:20-31`: a coarse histogram identifies a cut region,
keys above it emit directly, and crowded cut regions refine through 4096-bin
histograms before an original-ID tie. This is a related boundary-selection
design, not yet an exact field substitution for the two fixed digit schedules.
Its last-arrival barrier, parity-buffered histogram state and original-score
payloads remain source-specific realization obligations. Short decode rows use
`decode_select_row.cuh` with register-held scores.

Keep fixed graph bounds, per-device paged scratch exclusivity and ascending
ragged versus unordered paged presentation from the
[scope trace](trace-score-indexing-scope.md). Complete the resident/coarse-region
protocol dispositions alongside the remaining MLA backends and general top-k
family; do not duplicate one rule per shape class.
