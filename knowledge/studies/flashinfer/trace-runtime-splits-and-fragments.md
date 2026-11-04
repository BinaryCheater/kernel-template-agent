# Runtime split capacity and sparse block/page refinement

All source paths refer to the retained [tree](inputs/sources.json). This pass
continues `flashinfer/attention/prims_ts` through its typed runtime split
consumers and direct/materialized sparse metadata routes.

| Observable need | New candidate | Useful choice |
|---|---|---|
| Live lengths change but graph/workspace topology should stay reusable | [runtime_partition_capacity@1](delta/runtime_partition_capacity@1/template.yaml), [proof](delta/runtime_partition_capacity@1/proof.yaml) | Capacity determines addresses/participants; a derived useful prefix determines produced states and safe merge loads |
| Selected sparse blocks do not match physical cache pages | [semantic_block_page_fragments@1](delta/semantic_block_page_fragments@1/template.yaml), [proof](delta/semantic_block_page_fragments@1/proof.yaml) | Refine at gcd(block,page), resolve every fragment independently, precompute locators or supply them through a direct index view |

## Runtime useful work is not physical capacity

The split-KV helper is
`flashinfer/attention/prims_ts/kernels/fmha_decode/fmha_decode_resources/helpers_kv_tile_idx.py`.
For live tile count T, requested fanout R and instruction group I, :80-112
chooses L=max(I,ceil(T/(R*I))*I). :229-275 derives A=ceil(T/L); only s<A
has useful tiles. Aligned ranges cover[s*L,min((s+1)*L,T)). The two-split
specialization recovers A by comparisons against0 and L rather than runtime
division; it has the same range identity.

| Source choice | Candidate field / obligation | Binding |
|---|---|---|
| `decode.py:1456-1474` | `capacity`, `requested_splits`, `capacity_addressing` | Source retains requested fanout separately. Noncluster compact-grid GMEM uses the physical SM bound as prepared capacity; cluster or grid-limit paths retain their exact prepared topology. This is not a universal one-count-per-JIT recipe. |
| Helper :133-190 | `pre_partition` | Causal grouped-Q union bounds use the latest active Q row on the right; sliding-window left endpoint rounds down to a full tile and retains per-row masks. An aligned scheduling span does not widen logical row support. |
| `fmha_decode_kernel.py:2839-2880` | `inactive_protocol=retire_local`, `pre_retire` | Resource initialization and PDL acquire happen before count-based pruning. Inactive producer CTAs execute the explicit TMEM/dependent-release retirement path instead of blindly returning before setup or waits. |
| `reduction.py:615-685` | `pre_publish`, `capacity_addressing` | Useful s<A are cyclically assigned across the **fixed** reducer cluster. Loads are guarded before dereferencing partial O/LSE. Inactive caller scratch can be stale or uninitialized; masking a loaded NaN afterward is not the contract. |
| `fmha_decode_config.py:1665-1688`, runtime reducer topology | `inactive_protocol=participate_neutral` | Prepared reducer participants/capacity do not shrink to A. Inactive ranks retain collective/teardown responsibilities; locally retiring them would require another topology proof. |
| `mla_decode/throughput_latency_1cta/parallel_reduction.py:45-64,80-103` | Count windows versus capacity stride | Native reducer uses short runtime-count windows and16-split chunks, while partial-row addressing uses cfg.num_ctas_per_seq_kv. Unroll window, active count, allocated capacity and cluster residency are distinct quantities. |

reduction_partition@1 supplies exact-cover/merge algebra, but its no-empty-group
premise and one num_groups field do not encode fixed capacity versus live count
and synchronization participation. The new rule preserves all three roles.
It also requires count/alignment arithmetic and exact-division metadata to fit
and agree. Merge-state representation remains explicit: normalized O/LSE and
raw numerator/max/mass use different physical decoders.

T=0 implies A=0 in the split path. No partial may be read, and the route's
output policy decides whether it stores an empty result or skips inactive rows.
Do not infer a universal zero-fill policy from a standalone dense example.
Workspace/control reset and concurrent ownership remain separate native bindings.

## Semantic sparse blocks may cross storage pages

`q_token_kv_block_sparse_metadata.py:150-166` derives F=gcd(B,P), where B is
semantic block size and P is physical storage-page extent. Valid locators have
maximum at most2^31-1; the **count** may reach2^31. Negative values are reserved
for invalid entries. Metadata shape formulas at :924-982 use B/F fragments per
selected semantic block, not one locator per block.

Representative map, with original selected block b and fragment f:

1. t=b*B+f*F; logical page=t//P; subpage=(t%P)//F.
2. Resolve h=table[request,logical page] independently for that fragment.
3. Encode locator=h*(P/F)+subpage.
4. Consumer decodes physical page and subpage, then loads F consecutive rows.

| Source decision | Candidate field / obligation | Binding |
|---|---|---|
| `kernels/fmha_decode/direct_sparse_metadata.py:196-249` | `locator_mode=on_demand` | Direct metadata view resolves selected block and fragment before physical page translation. Negative/out-of-prefix block IDs become invalid; required public prefixes must still satisfy their contract. |
| `q_token_kv_block_sparse_metadata.py:1518-1560`; `include/flashinfer/attention/prims_ts/q_token_kv_block_sparse_metadata.cuh:254-295,296-345` | `locator_mode=precomputed` | Q1 materializes direct fragment rows; grouped routes build union/membership rows upstream. The CUDA Q1 writer initializes the entire locator capacity to-1 because speculative staging can read beyond the live prefix. |
| `fmha_decode_resources/smem_resources.py:180-227` | `locator_roundtrip`, `pre_sentinel` | For power-of-two subpage ratios arithmetic shift preserves page=-1 and a bounded offset. Other ratios first guard locator>=0 before division. C++ truncating division of-1 by3 would otherwise yield page0. Flat cache head-offset addition also preserves the invalid sign. |
| Same file :1531-1548 | `pre_lifetime` plus indexed_gather@1 prefetch choices | Held locator spans remain live through K/V and feature-stage consumers; data-buffer stages advance independently. A locator buffer cannot be recycled merely because one K load issued. |
| Public wrapper :677-690 | `pre_support`, `pre_padding` | Selected prefix IDs are distinct and in range; holes are not a general compaction feature. Full-fragment V padding must already be finite zero/admitted neutral. Causal score masking does not sanitize NaN/Inf V. |
| `decode.py:1420-1453` | `pre_consumer` target/route guard | Encoded-subpage FP8 source profile may reject direct, inline and cluster publication, selecting a separate split reducer. Valid map arithmetic alone does not qualify every native pipeline. |

For B=32,P=24,F=8, one semantic block spans three fragments in physical page31
and one in physical page7 under the checked table. Physical token744..767
is followed by168..175, not768..775. Assuming page adjacency silently loads
another value. P=12/20/24 cases also exercise non-power-of-two decoder ratios.
The source supports semantic blocks4/8/16/32/64/128 and storage pages that are
positive multiples of4; no arbitrary target/extents are admitted implicitly.

indexed_gather@1 handles transfer through an established index map. It does
not derive the two extents, their gcd or encoded page/subpage units; that is
what the new refinement supplies. grouped_sparse_union_masks@1 can provide the
upstream union/membership choice, with its set-semantics premises preserved.

### Current-block policy is semantic input

For zero-based query p, bound=(p+int(count_current_block_in_topk))//B.
Selected prefix length=min(topk,bound), with IDs in[0,bound). Tail length is
p+1-bound*B. True permits a just-completed current block to be among selected
blocks and appends only an incomplete tail; False selects historical blocks
and always appends the current block, including its completion token. Preserve
that workload option when materializing or directly resolving fragments.
No per-mode Template is needed once its exact support is bound upstream.

## CLC scheduler: retained ownership boundary

`mla_decode/throughput_latency_1cta/kernel.py:315-370` allocates two16-byte CLC
response slots and delegates fetching to the CuTe DSL
`ClcDynamicPersistentTileScheduler`. :413-463 adds a two-stage load-to-scheduler
throttle and a two-stage,16-byte CLC response publication edge to all CTA threads.
`tasks.py:1521-1568` still advances the queue past runtime-empty packed-Q tiles,
while those tiles publish no load token. Empty computation is not queue termination.

This is source-backed dynamic work-queue plumbing, not an exact binding to
persistent_tile_scheduler@1's atomic increment program. Native cancellation,
response decode and progress are delegated library obligations; dependency bodies
are not retained here and are not fetched. Static persistent and CLC modes are
mutually exclusive. The1CTA CLC queue does not enumerate KV splits; its source
scope requires split_kv=1. The2CTA split-aware queue is another realization.

## Evidence and next scope

[Logical checks](checks/check_runtime_capacity_and_fragments.json) cover3072
aligned partition cases, changing live counts inside one capacity, poison/stale
inactive scratch, fixed collective participation,324 block/page mappings and60
current-block boundary cases. Integer range and sentinel witnesses distinguish
count versus maximum locator and guarded versus truncating division. These
checks establish map/state invariants; native PDL/TMEM/cluster/TMA publication
remains explicitly outside them.

The task-scheduled split and sparse-map design choices now have dispositions.
Continue remaining B12x/model and specialized training routes, plus the named
orthogonal-preconditioning/ExpCast and native queue gaps, then advance through
GEMM and the remaining full-library inventory. Attention and library closure
are not inferred from these two representatives.
