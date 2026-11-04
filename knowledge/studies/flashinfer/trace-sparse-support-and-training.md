# Sparse support preparation and attention training

Source: retained [revision](inputs/sources.json). The
[selected generated-source map](inputs/training-source-bindings.json) gives
revision-relative paths for the index-preparation, DSA NoPE backward and GQA
backward representatives. Public Cake wrappers reach retained CUDA through
their package `build.json`; prebuilt/JIT loading is an implementation route,
not proof of runtime execution in this study.

## Sparse support is an occurrence list

`flashinfer/experimental/cake_dsa_train/cake_backend.py:1188-1300` transforms
per-document key IDs into global rows. Original slot validity includes index
bounds, optional caller position limit and causal key bound
Lk-Lq+local_query. Query/key segment lengths have independent prefix arrays.
`_slot_positions` at :1367 is one-based, so its <=length test means the
zero-based original slot is <length, not an extra included element.

[stable_sparse_support_compaction@1](delta/stable_sparse_support_compaction@1/template.yaml)
and its [proof](delta/stable_sparse_support_compaction@1/proof.yaml) capture
the choice between retained holes and stable valid prefixes. The generated
index-preparation source, :80 onward, clamps the original active input span,
then scans original chunks; the Python twin uses a stable validity sort.
The two representations need different lengths: last valid position+1 versus
valid count. A later position cut must not be applied to packed ranks.

Forward/backward consume the same prepared global rows and lengths;
`cake_backend.py:5190-5205` states this at the autograd boundary. Repeated IDs
are separate occurrences. This is neither shared_index_union@1's sorted unique
two-consumer union nor first_occurrence_record_scatter@1's winner election.
The latter resolves duplicate **write destinations**; it must not deduplicate
an attention support list.

The [logical check](checks/check_support_compaction.json) enumerates7737
slot/cut combinations with holes, out-of-range IDs and duplicates. It preserves
original-slot metadata and additive scatter multiplicity. For row[-1,2,2,1]
with original limit2, cut-then-pack keeps[2]; pack-then-cut incorrectly keeps
[2,2]. Equal-score support[2,2,1] with values[9,9,0] returns6, whereas dedup gives
4.5. An unequal query/key segment witness checks local-to-global causal IDs.
Native prefix scans, metadata publication and changed tiled rounding remain
implementation obligations.

## DSA training: concrete rule composition

Public `flashinfer/dsa_sparse_attention.py:118-162` defines absorbed scores
Qlatent*C^T+Qrope*Krope^T and output P*C. K=V in the latent space, so its
backward includes both score and value derivative paths into the same C.
Geometry is 64 heads, latent512 and rope64; the NoPE relative omits the rope
term and its derivative. Indices are global rows, invalid entries can occur
anywhere and an optional topk_length cuts positions before preparation.

| Decision / source binding | Existing rule and boundary | Disposition |
|---|---|---|
| `experimental/cake_dsa_train/README.md` forward/autograd contract retains BF16 O and BF16 residual from the same FP32 result | rounding_residual_checkpoint@1 | Residual reconstruction is admitted arithmetic accuracy, not a general lossless FP32 codec. Both checkpoints retain query/head/generation identity. |
| DSA NoPE backward source :416-427 reconstructs P with exp2(score*scale_log2-LSE2) | softmax_probability_recomputation@1 | Saved normalizer, mask/support, scales and exact prepared occurrence list must match forward. Discrete selected indices themselves receive no gradient. |
| Same kernel signature uses delta, O/o_lo and dO; package backward preprocess forms the row center | softmax_gradient_row_checkpoint@1 | Center consumes the declared reconstructed O and dO; no incoming-LSE derivative support is inferred from a forward return_lse switch. |
| Same kernel :1149-1177 drains FP32 vector red.global.add contributions through optional destination mapping | indexed_additive_scatter@1 `vector_width=four_elements` | Preserve every repeated support occurrence and every duplicate destination-map contribution. Vector atomicity is per element, not one record transaction. Valid destination-map range/alignment is a caller premise. |
| Package README direct-accumulation section: natural-layout main-stage variant versus permuted FP32 intermediate then cast/add | physical_storage@1 plus additive scatter; output-layout fusion choice remains a binding | Direct path removes intermediate accumulator/zero/cast stages but adds a lane transpose. Source size policy is a heuristic, not measured benefit here. Caller-owned += buffers retain initial contents and are not zeroed by each pass. |
| README key-range passes; backend prepares bwd_compact/main_pass stages | reduction_partition@1 and consumer_complete_scratch_chunking@1 only for its lifetime contract | Key-range passes preserve every occurrence, carry dQ FP32 partials and complete all passes for one token chunk before scratch reuse. Default policy treats multi-segment workloads separately; no uniform L2 benefit is inferred. |

A pass is a key partition, not a new independent softmax row: every reconstructed
probability uses the full forward row normalizer. Latent dC equals dC_from_K
plus dC_from_V. Selecting only one derivative branch or normalizing each key
pass independently changes the operator. Native MMA rounding, atomic order,
permuted-to-natural output drain and pass-specific dQ association remain
source-bound numerical obligations.

## Packed causal GQA training relative

`flashinfer/gqa_train.py:17-40` routes to `experimental/cake_gqa_train` for BF16
head256 on SM100/103/107, even Hq/Hkv and bottom-right causal alignment.
The package backend at :534-609 launches bwd_pre -> selected main mode ->
bwd_post. The selected backward CUDA signature at :322 takes Q,dO,K,V and
saved preprocessing state. The source package README's kernel table accounts
for the ownership:

- Forward CTA pairs own256 query rows per segment/query head.
- Preprocess forms dot(O,dO), converts natural LSE to base2 and initializes
  FP32 dQ scratch; this binds softmax_gradient_row_checkpoint@1 without the
  DSA-specific residual pair unless one is explicitly supplied.
- Main CTA pairs own128 key rows and one KV head, visit grouped query heads
  two at a time, and accumulate dK/dV in FP32 TMEM. P/dS become BF16 only as
  MMA operands. This is grouped-head reduction ownership, not a different
  derivative identity.
- dQ uses additive partials and is scaled/cast in bwd_post. Main output modes
  are BF16 final stores, FP32 window stores, or mapped FP32 adds; their initial
  destination semantics differ and must remain explicit.

`max_seqlen_k` bounds the actual key grid and must cover every segment; a too
small value omits work, while an oversized bound may launch empty CTAs. The
empty-input branch preserves caller += destinations and zeros newly produced
gradients. Shared workspace cannot overlap live users. Native deterministic
claims are scoped per output: additive dQ/row-map updates do not inherit a
bitwise guarantee from exclusive dK/dV ownership.

## Block-sparse inference support relative

`flashinfer/sparse.py:45-112` converts BSR indptr/indices to VSA lists and counts.
For GQA packing factorR it repeat-interleaves each original query-block list R
times, then broadcasts across heads. Packed query-block b therefore reads
original b//R; repeating the whole list array instead would choose the wrong
support. Bind grouped_head_row_packing@1 metadata identity and indexed_gather@1
only after the selected VSA backend's block size, masks and count/padding
contract are bound. This helper uses host metadata conversion, unlike the
DSA on-device stable preparation; a shared word index does not imply the same
execution route.

This pass establishes the common training/support mechanisms and new stable
compaction choice. Deterministic indexer-training, sparse backward ownership
variants, quantized/model attention and VSA-specific physical masks remain
separate inventory work; no full-family closure or device validation is claimed.
