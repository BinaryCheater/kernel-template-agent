# Version1 projection and extended domain

The original [operator_fusion@1](../../../../seed/references/cutlass_ampere_tf32_gemm/templates/operator_fusion.yaml)
remains addressable. Map old consumer_kind, intermediate_site and fused_ops
unchanged; bind new checkpoint=none and workload checkpoint_bytes=0. Old
pre_tile_local implies the new disjunction, so its existing branches are retained.
No old field, effect term or obligation is removed. The old no-cross-tile and
same-values obligations retain their names; explicit lifetime/no-omission premises
must be checked when rebinding an old instance, rather than presumed proved.

New domain: dense_layer/activated_dense_layer require contract consumers and the
entire feature dependency resident locally; optional materialize retains a
checkpoint for an external consumer. The traffic formula adds checkpoint_bytes.
That value counts actual retained writes for this edge, zero if none, and excludes
future reader traffic accounted by those consumers. Storage remains a separate
physical_storage binding. Existing numeric/lifetime proofs must be reopened for
changed checkpoints, dense consumers or changed layer precision.

Source substitution for the historical linear_combination case keeps its old
program, no checkpoint and identical byte count. The old softmax_then_matmul case
keeps its rescalable locality branch. These are contract projections, not replay
of historical GPU executions. Full end-to-end replay remains deferred.
