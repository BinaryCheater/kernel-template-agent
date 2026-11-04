# spconv: reusable choices

Status: extracted by distinct retained NVIDIA design; candidate pending independent review.

Sparse neighbors reused across channels -> [offset-indexed incidence](delta/offset_indexed_sparse_incidence@1/template.yaml) -> choose explicit pair lists or an output-indexed map/mask. Preserve exact geometric edges and shared output identity. Dense submaps can reuse seed output_tiling@2; legal partitioned sums reuse reduction_partition@1.

Worked case: active sites 0 and 2 under a three-offset 1D padded stencil share output 1 through distinct offsets. A mask combines both bits; it must not collapse the arithmetic contributions. A missing center site leaves -1 and contributes no feature.

[Inventory](inventory.md) and [trace](trace.md) contain dispositions and retained-source references. Source substitution checks word/bit ownership (offset 33 -> word 1, bit 1) and competing contributors. No executable check was needed to resolve the mapping. Publication and actual generated-device behavior remain source premises, not proven results.

Open: external cumm hash/MMA bodies, quantized epilogue details, box suppression and average-pooling backward arithmetic. Point-to-voxel capacity truncation is recorded without promoting it to exact grouping. These gaps bound this pass; no implementation audit or GPU experiment performed.

Distinct-design continuation: [additional trace](trace-additional-designs.md) supersedes the initial shallow dispositions. Bounded grouping -> capacity_bounded_bucket_append@1; inverse-neighbor gradients -> transposed_incidence_owned_reduce@1; support-sorted convolution -> predicate_keyed_output_permutation@1. The pool entries have paired outlines.

Corrections: box operations and gather.py are CPU-only; do not count them as unread NVIDIA mechanisms. Max pooling is zero-seeded in the launcher; average backward multiplies by count and is not an established average gradient. cumm is version-ranged, not pinned: its exact MMA/quantized body is an unavailable dependency boundary, while the sparse scheduling, forward/backward incidence, masks, pooling, voxelization and epilogue organization are now concretely traced. Status stays active until final source reconciliation and candidate verification.

Source-study closure: the concrete continuation supersedes initial unread dispositions. Local candidates validate, retained sources reopen, and their identities do not conflict with other study candidates at reconciliation. Device realization/proof and explicitly identified external dependency boundaries remain open; no major retained local design is closed from a directory listing alone.
