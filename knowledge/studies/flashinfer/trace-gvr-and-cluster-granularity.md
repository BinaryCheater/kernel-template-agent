# GVR count certificates and cluster granularity

When a batch mixes long and short rows, use a full cluster for each expensive row
and pack independent short rows one per CTA in that same launch shape. The new
[cluster_granularity_switch@1](delta/cluster_granularity_switch@1/template.yaml)
and [outline](delta/cluster_granularity_switch@1/proof.yaml) preserve row identity,
collective domains and prepared metadata. GVR2's sample-estimated thresholds are
a separate exact-selection organization with count-certified fallback.

Paths below refer to the retained [tree](inputs/sources.json).

| Source decision / binding | Rule field or reuse boundary | Evidence |
|---|---|---|
| `flashinfer/topk_varlen/kernels/gvr_topk_decode_lb.py:110-165`: classify floor(seq_len/compress_ratio)>long_threshold, stable long/short request partition | `cooperate_threshold`, `pre_keys`, `pre_classify` | Threshold units are scan entries; compression1/4 is workload representation. Prepare covers every live request and zeroes inactive prefix-scan contributions |
| `:269-348`: cluster g<L*next_n cooperates on one expanded long row; later cluster/rank slots cover short rows | `cluster_ctas=2/4/8`, `rows_per_request=next_n`, exact `program` map | Counts L/S describe requests. Multiplication by next_n occurs in the consumer; counters are not already row counts |
| `:183-196,326-348`: long branch is cluster-uniform; short body compiled at cluster_size1 has no peer synchronization | `pre_cluster`, `pre_independent` | Partial short clusters may skip inactive ranks because no surviving short CTA waits for them. Cooperative rows still need all ranks and matching retirement |
| `:350-389`: prepare once per decode step and reuse across layers; launch max_batch_size*next_n*cluster_size CTAs | `max_requests`, `pre_metadata`, launch effect | Fixed-capacity surplus clusters still launch but do no row work. The request-only partition is independent of next_n; a changed expansion needs a matching consumer/layout, not inherently another partition |
| `gvr_topk_decode.py:2780-2820`: original row=req*next_n+local; valid length=floor((seq_len-next_n+local+1)/compression), clamped to[0,physical_width] | `pre_expansion`, unchanged row contract | Classification uses request scan key; it must not replace each expanded row's actual valid bound. Previous-step hint IDs shift by local+1 only for uncompressed representation |
| `gvr2_topk_decode.py:20-48,1436...`: one sampled histogram calibrates a threshold ladder; main/streaming, register, cluster and register-cluster families share the certificate goal | Existing count-certificate obligations, with source-specific ladder/transport | Sampling affects expected work, not authorization to omit winners. No source timing comments are measurements in this study |
| `:2667-2736`: full histogram count-crossing total>=k accepts a rung; otherwise retries staged thresholds, gathered anchor/floor, then degeneracy fallback | `count_certified_topk_filter@1.pre_filter/pre_append/pre_select` supply completeness/fallback obligations | Existing v1's one candidate list is not the entire multi-rung source state. Its certificate discipline applies only after binding every counted domain, capacity and fallback generation |
| `:2736-2793`: valid crossing and crossing-bin count drive refinement; complete list flag guards staged candidate reads | Completeness and storage are separate facts | Enough qualifying count does not prove every record fits a staged buffer. Incomplete staging must not be consumed as a full input |
| `:2886-2914,3011-3026,3164-3190`: key-space narrowing over a crossing list, a complete staged candidate set, or original whole-row window | Same exact radix threshold/tie goal with explicit input source | Degenerate sample, oversized crossing bin or exhausted rung cannot silently turn into truncated selection. Whole-row fallback keeps prefill lead/window exclusions consistent |

The [logical mapping check](checks/check_cluster_granularity_switch.json) covers3048
class-pattern, expansion, cluster-width and capacity cases, including zero live
requests and partial short tails. Each original request/local-row pair has exactly
one body owner; cooperative rows have all ranks, independent rows have one CTA.
A single short row in a four-CTA cluster is safe only if it introduces no
four-party cluster barrier or remote read. The check also distinguishes raw
length classification from the source's divided scan key.

The same prepared request partition can be consumed with different row expansions
when the selected kernel and physical tensor layout agree. Reusing a table after
the request population shrinks, or binding the wrong expansion to a tensor,
violates a different premise. Keep request-level metadata generation separate
from per-layer logits and per-consumer expansion. Source fixed capacity is an
allocation/graph-capture constraint, not a claim that inactive CTAs disappear.

GVR2 register and cluster families also preserve degeneracy guards named in the
module's source notes: incomplete count crossing and infinite sampled bracket
widths require key-space or whole-row escape. Existing finite-score radix
proposals do not automatically extend to infinities, NaN, signed-zero tie classes
or every native tie-emission order. Those are explicit key/numeric realization
bindings. Native selector correctness, DSM progress, compiler resource allocation
and all rung/candidate lifetimes remain their stated obligations.

The scheduling delta is distinct from `two_family_cta_pool@1` ticket claims by
individual CTAs and `cost_partitioned_worklists@1` cost-balanced preassignment:
neither binds a cooperative cluster versus independently active ranks under one
physical cluster width. Recurrent, MoE, model-block and communication families
remain next in the [full inventory](inventory.md); native sampling/selection gaps
stay linked in the [sampling inventory](sampling-inventory.md).
