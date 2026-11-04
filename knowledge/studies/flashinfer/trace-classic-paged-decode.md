# Classic paged decode: coordinate and state composition

When one query position attends paged KV, classic decode distributes feature
dot products across x lanes, grouped heads across y and key partitions across z.
The retained [source](inputs/sources.json) has the same broad mechanisms as the
older seed, but current planning, empty-state and tail contracts are bound here.
Paths below are relative to that source tree.

## Selected CUDA-core representative

Take SM80, FP16 Q/K/V, D128,8 query heads,2 KV heads, no rotary or custom
score changes, and the CUDA-core path selected with use_tensor_cores=False.
csrc/batch_decode.cu:38-80 checks equal QK/VO dimensions and builds the plan;
the run path :130-240 binds its workspace offsets into Params. JIT config and
instantiations lead to BatchDecodeWithPagedKVCacheDispatched at
include/flashinfer/attention/decode.cuh:756-838. Feature/head/key ownership is:

- vec_size=max(16/sizeof(KV),D/32)=8; bdx=D/vec_size=16.
- bdy=GQA group size4; num_threads=max(128,bdx*bdy)=128; bdz=2.
- tile_size_per_bdx=1; num_stages_smem=2 for compute capability major>=8.
- thread(tx,ty,tz) owns query head4*kv_head+ty and feature8*tx+i.
- iteration t loads keys chunk_start+8*t+4*tz+ty; each query head reads all
  four staged keys in its z partition. The two z states combine before output.

| Decision / source | Existing rule and field, or gap | Concrete binding / invariant |
|---|---|---|
| decode.cuh:62-118,424-438 | reduction_partition@1 plus cross_lane_reduction@1 lanes16,result_to all_lanes,op add | The x half-warp reduces eight feature products per lane. All16 lanes remain converged; y heads do not mix. Key partitions are z's disjoint four-key groups, not separate query heads. |
| page.cuh:185-235 | indexed_gather@1 granularity page; index_prefetch smem_batched,index_reuse16 | Request logical token uses page_iter=indptr[request]+token/page_size and entry=token%page_size, then indices[page_iter] chooses the physical page. The physical page/head/token strides distinguish NHD and HND. Empty requests have length0 even with no last page. |
| decode.cuh:490-547 | indexed_gather@1's cooperative offset table, with current generation map | x lanes precompute future token offsets rather than merely their own feature address. One table covers16 iterations; per-thread vector loads add8*tx to the selected token base. Refresh occurs when (iteration+stages)%bdx==0, for the next horizon. Static assertion requires stages<=bdx. |
| page.cuh:228-235 plus decode.cuh:515-592 | boundary_predication@1 runtime_length/token_mask | Protective lookup checks page_iter against the global page-list length and returns placeholder offset0 otherwise. It does not validate a request's logical token range or physical page ID. Transfer predicates still enforce chunk bounds, and actual page IDs must be allocated. A pointer fallback is not permission to read another request's KV. |
| decode.cuh:515-592 and cp_async.cuh | boundary_predication@1 fill none for K,zero for V; async_transfer/multi_buffer_rotation@2 | K retains untouched residue storage because the score is explicitly replaced by -inf after QK. V must be zero-filled: zero probability times stale NaN would contaminate O. Two data stages issue distinct K/V copy groups; wait_group<3> plus CTA joins delimit reads/refill. Native instruction/group ordering remains a binding obligation. |
| decode.cuh:441-454,600-609 | physical_storage@1 buffer shared,share_order this_first | After all KV reads and wait_group0, shared K/V storage becomes FP32 partial O, while the offset table becomes m/d storage. Allocate max(KV bytes,partial-O bytes), not just the earlier narrower payload. For this binding KV8192B exceeds O4096B, followed by max(offset1024B,stats64B), total9216B. No live offset reader may survive into statistic overwrite. |
| decode.cuh:157-188,600-620 | smem_partial_combine@1 topology serial,combine_op rescaled_merge,result_to all_threads | Two z-partition groups store their numerator fragments and m/d, synchronize, then every reader combines both complete states. Only z0 writes final output. This is output-state merging, not the prefill feature-owned PV exchange. |
| decode.cuh:90-118,state.cuh:39-93,variant_helper.cuh:84-96 | online_rescaled_reduction@2 for positive finite mass; empty_support_normalization@1 after explicit representation mapping | Base2 state carries reference m, mass d and unnormalized numerator o. Score scale includes log2(e). Empty initialization(m=-inf,d=1,o=0) is tagged empty, not raw positive semantic mass. compute_qk clamps the exponent reference to a finite lower bound, empty merges suppress -inf-minus--inf NaNs under the native max contract, and output checks the empty tag. Map tagged empty to semantic zero mass before applying the old normalization rule. |
| decode.cuh:756-838,cascade.cuh:374-475 | split_combine_kernel@1 rescaled_merge,persistent,f32 for selected FP16 input/FP32 split workspace | Main kernel writes normalized partial output and base2 LSE; second kernel merges them at o_indptr ranges. A later stage must not treat normalized O/LSE as raw numerator/m/d. Empty merge ranges write zero/-inf; one state copies. PDL paths include an input wait and completion launch marker; actual dependent-launch visibility remains a native obligation. |

The [logical check](checks/check_classic_attention.py) follows request1 token35,
head1,feature61 through page size16, indptr[0,3,8], pages[7,1,9,6,2,8,0,5].
Its packed position is83, so page-list entry5 selects physical page8 at entry3.
Element offsets are33725 for NHD and35261 for HND. Dropping the request's
indptr term selects a different page. The same check follows34 prefetch
iterations across two horizon refreshes. [Observed results](checks/check_classic_attention.json)
are coordinate/algebra evidence, not CUDA execution.

## Planning and empty work

include/flashinfer/attention/scheduler.cuh:64-97 estimates a pages-per-chunk
value using occupancy and binary search; this is a workload heuristic, not a
semantic constant. :349-365 creates request_indices,kv_tile_indices,o_indptr,
with at least one chunk even for a zero-page request. Counts[3,5,0] at two
pages/chunk give chunk counts[2,3,1], request IDs[0,0,1,1,1,2], tile IDs
[0,1,0,1,2,0], and o_indptr[0,2,5,6]. At page size16, request1 with73
valid tokens uses ranges[0,32),[32,64),[64,73). Its final physical page
padding is excluded independently from the page-list allocation bound.

Plan metadata :434-500 reserves stable workspace addresses and graph-padded
launch slots with block_valid_mask. In decode.cuh:424-435 a padded slot may
load its allocated metadata entry but returns uniformly before using it to
dereference a request's KV or reaching CTA barriers. Valid zero-length work
still owns a zero output/-inf statistic. This differs from simply omitting an
output write. empty_work_pipeline_projection@1's no-fake-payload principle
applies, but this one-shot CTA exit does not need a persistent pipeline epoch
transition. Workspace-size queries do not themselves compute attention.

Single-request CUDA decode at :676-749 shares the core organization but its
thread-count heuristic treats GQA8 specially (512 half-precision or256 one-byte
threads), unlike batch decode's max128 rule. Optional tensor-core decode routes
through a single-query prefill representation in flashinfer/decode.py:1275
onward; do not claim it executes this CUDA-core dot-product kernel.

Default score variants in variants.cuh:32-98 bind custom masks, ALiBi,
sliding-window coordinates and pre-mask softcap. These alter semantics, not
the page-address rule. RoPE can take an explicit per-request query position;
otherwise decode uses the last KV position. Scalar/metadata scales and the
non-softmax add path need their own operation binding. Kernel-specific target,
numeric and source-stage restrictions stay distinct from mathematical reuse.

No new rule is needed for this classic decode representative. The hardest
reuse, indexed_gather@1, is concretely supplied by its cooperative shared
offset table and exact coordinate checks. Tagged-empty decoding and native
prefetch/copy visibility remain explicit implementation boundaries rather than
an invented normalization-state interface.
