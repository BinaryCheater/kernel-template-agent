# MQA score indexing: scope and first reuse bindings

These entry points produce/select scores, not normalized attention. Source
paths below are relative to the retained [tree](inputs/sources.json). This is a
scope/reuse trace; selection/device relatives remain open and have not been
counted as a closed family.

| Entry and observed contract | Existing rule binding or gap | Next distinct-design work |
|---|---|---|
| `flashinfer/dense_mqa.py:47-84`: E4M3 Q[Q,H,128], key codes plus per-key FP32 scales, signed FP32 head weights; row window [ks,ke), outside cells written -inf | nonnegative_scale_hoisting@1 permits a nonnegative key scale shared across head reduction, even with signed head weights; block_scaled_partial_accumulation@1 binds grouped scales where present | Follow the prepared dense backend's actual fused consumer and score rounding. Its route catalog restricts shapes and has no fallback. Do not infer runtime support from the public signature alone. |
| `flashinfer/paged_mqa.py:7-30`: default metadata entry emits an unused placeholder; optional prepared schedule uses a real table | No runtime_work_descriptors@1 claim from placeholder bytes | Follow the selected consumer before treating a metadata object as a scheduling plan. Per-page codes/scales need matching page/row generations. |
| `flashinfer/sparse_mqa.py:7-55`: prepare sparse metadata, then compressed logits with aligned windows and packed MXFP4/MXFP8 inputs | shared_index_union@1 is available only after confirming its two sorted unique consumers, compatible physical mappings and independent output slots; indexed_gather@1 handles subsequent payload loads | Verify the actual metadata producer and its caller validity/tail contract; an API named metadata does not prove the old two-consumer format. |
| `flashinfer/dsa_indexer.py:35-76,129-194`: BF16 training score sums signed head weights times ReLU(scaled dot); exact top-k order is score descending then larger local key ID, signed zeros equal; outputs ascending IDs with -1/-inf padding | nonnegative_scale_hoisting@1 binds only positive shared scales and admitted arithmetic. sorted_run_topk@1 is not a direct binding: its k<=32 and smallest-ID tie order differ; new threshold/radix candidates are required | Preserve independent query/key segments and visibility floor((offset+u+1)/ratio), including negative offsets and empty support. The host's bitwise claims require inspection of the actual fixed score/reduction implementation. |
| `experimental/sm120_dsa_indexer/README.md`, `_api.py`: resident-row score/select overlap; long prefill sampled threshold, candidate append/exact selection, exact rescan when candidate bounds fail; decode scan then histogram/register selection | [Count-certified filter and radix boundary rules](trace-exact-indexer-selection.md); coarse-region decode realization remains open | A threshold is a performance heuristic only if the acceptance predicate proves all top-k winners are retained; overfull/truncated candidates require exact fallback, not approximate truncation. Inspect the C++ producer/consumer before proposing a guarantee. Epoch-tagged completion may explain reset-free state once its actual counters/readers are bound. |
| `flashinfer/dsa_sparse_attention.py:118-162`: downstream absorbed MLA takes global key rows, invalid slots anywhere and optional position cut | indexed_gather@1 plus predication; training scatter/checkpoint rules remain available | Indexer returns segment-local IDs; conversion to global rows must add the correct segment base before this consumer. Do not silently pass one representation as the other. Retain slot multiplicity unless the selected producer/consumer explicitly deduplicates. |

SM120 serving uses FP8 or MXFP4, head counts 32/64 and key dimension 128;
query/head scales are folded into weights while per-key scales have their own
binding. Ragged and paged selection have different output order contracts:
ragged ascending IDs versus paged selected set in arbitrary order. The sorted
run rule cannot impose its output presentation on every consumer. Fixed graph
launches follow shape bounds, not current lengths. Shared per-device paged
scratch requires nonoverlapping invocations and survives captured consumers;
source performance tables are not measurements from this study.

The [long-row trace](trace-exact-indexer-selection.md) now extracts the complete
count certificate and two radix schedules. Continue resident/coarse-region
protocol bindings, then reconcile dense,
paged and sparse score layouts with DeepGEMM's actual pinned rule fields. Keep
training, score selection and sparse-attention arithmetic as separate objects.
