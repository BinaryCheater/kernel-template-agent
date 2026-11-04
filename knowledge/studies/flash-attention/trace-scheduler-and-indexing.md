# Work assignment, grouped rows and paged operands

Sources below start at flash_attn/cute/ in the [pinned tree](inputs/sources.json).
Work coordinates, physical CTAs and operand addresses are distinct objects;
one arithmetic identity cannot establish all three mappings.

| Source choice | Rule/field or gap | Concrete binding / boundary |
|---|---|---|
| pack_gqa.py:15–36: group query heads by shared KV, fold within-group heads into query rows | [grouped_head_row_packing@1](delta/grouped_head_row_packing@1/template.yaml) | Packed row r=t*g+u, KV head h//g, original query head=kv*g+u. Q/O and LSE have different head-axis positions and need separate stride bindings. |
| pack_gqa.py:63–108: sparse MLA pads physical head tiles using dynamic head-first TMA bounds | Additional guarded implementation; not the complete-group rule alone | TMA zero-fills fake Q heads/drops their stores. Non-TMA LSE/sink/row_max require explicit real-head guards because hierarchical modulo coordinates can wrap a fake head into the next token. |
| pack_gqa.py:109 onward: pack/unpack TMA coordinates to retain descriptor rank | Physical layout/rank binding; tensor_axis_permutation@1 alone is insufficient | Split-and-fold semantics remain the new grouped-row rule; avoiding a fifth TMA mode is a backend representation choice, not a different query-head grouping. |
| paged_kv.py:17–41,165–202: logical token plus left padding → page-table quotient/remainder → physical page/offset | indexed_gather@1: granularity page,index_prefetch registers,index_reuse1; new native realization | Invalid logical rows suppress page-table reads and use placeholder page0; that placeholder is not permission to read its payload. Validity also gates the later transfer. |
| paged_kv.py:216–275: 128-bit cp.async gather by row, K/V head dimensions and orientation | vectorized_access@1 + async_transfer@1, subject to matching instruction/zero-fill realization | SM90 V uses token×value view; SM100 V uses value×token. Page size need not equal compute tile size in this gather branch. Row/dimension tails remain predicates; native predicated-copy behavior is not inferred solely from a valid pointer. |
| interface.py SM90/SM100 construction; MLA frontends | Transfer selection boundary | Regular TMA path uses its admitted page/tile relation; mismatching page sizes select cp.async manager. MLA kb64 also supports whole64-key page blocks through TMA. A page index by itself does not establish tile containment or transfer extent. |
| tile_scheduler.py:252–375,508–790: SingleTile and L2-sectioned LPT coordinate maps | output_tiling@2 custom index_map; exact scheduler map is an instance | Static nonpersistent grids assign one work item, causal/local LPT reverses/reorders M. A while-shaped consumer API does not make every scheduler persistent. |
| tile_scheduler.py StaticPersistentTileScheduler: cluster-stride progression | persistent_tile_scheduler@1 static_stride concept, cluster coordinate realization gap | SM100 counts logical work tiles in clusters; physical grid x=C*cluster_slot+cta_rank. Work_m is shared by CTAs of one cluster, not a physical CTA id. Static progression strides by launched cluster count. |
| tile_scheduler.py:1390–1440,1530–1605: varlen decoder and semaphore fetch | persistent_tile_scheduler@1 atomic_counter concept; segmented decoder binding | Flat tile id is decoded through per-batch block prefixes, head/split extents and optional virtual batch order. Dynamic count is packed in high16 bits of split_idx; consumers unpack it before partitioning KV. Field-width/range bounds remain required. |
| tile_scheduler.py:675–708, SchedulerState: CLC work adapter | Hardware work-claim realization gap | TwoCTA claimed x divides by cluster size before logical decode; invalid claim coordinates are not trusted. Prefetch/wait/release uses a published work-info stage. Native hardware uniqueness and cluster participation remain external backend obligations. |
| prepare_scheduler.py:35–68,197–317: per-sequence dynamic split count and metadata publication | reduction_partition@1 explains splitting only after a legal count is fixed; heuristic and publication are instance bindings | One-wave estimate caps splits at static allocation and optionally preserves at least one split for zero-fill. No-op sort TODO does not establish a sorted virtual batch map. Consumer must wait for matching metadata, including PDL early launch. |
| block_info.py:35–68: split interval covers tile-local KV range | reduction_partition@1 contiguous groups with explicit empty groups | For5 blocks and3 splits, ceil group2 gives ranges[0,2),[2,4),[4,5); with6 splits last range is empty. All load/MMA/softmax/correction roles must agree on that emptiness. |

Index substitution: page size96,leftpad5,logical token100 maps to quotient1,
offset9. With page table(7,2,...), the physical token is page2,offset9. A128-key
compute tile starting at logical0 crosses physical pages; one descriptor/page
lookup cannot represent the whole tile. This is distinct from a128-key page
boundary even if both happen to share the same total allocation.

Grouped-row substitution: g3,token2,query head7 maps to KV head2,within-group
head1,packed row7. A causal predicate must use token2, not7. For work_m2,
q_stage2,twoCTA, logical Q block indices are8/9 for stage0 and10/11 for stage1,
as defined by flash_fwd_sm100.py:128–134. A physical launch index cannot replace
this work coordinate in a persistent/claimed schedule.

These source groups are accounted for by concrete reuse and named backend
mapping/publication gaps. No native address/claim/progress check was run. The
source metadata heuristic is not a measured optimum and an unimplemented sort
is not pursued as an unrelated host fix.
