# NCCL reusable choices

Status: extracted; pending independent review. Final library in Worker E queue.

| Kernel need | Rule and useful choice |
|---|---|
| Replicated reduction over a ring or tree | [Ring phases](delta/ring_reduce_scatter_allgather@1/template.yaml) / [tree subtree ownership](delta/subtree_reduce_root_broadcast@1/template.yaml): one contribution per rank, final post-op once |
| Local peers plus network heads | [Hierarchical head stripes](delta/head_partitioned_hierarchical_collective@1/template.yaml): stripe ownership and consistent registered address permutation |
| Reusable remote staging or direct buffers | [Ready/retired credits](delta/credit_gated_circular_transfer@1/template.yaml), [direct descriptor exchange](delta/acknowledged_direct_buffer_exchange@1/template.yaml), [tagged lines](delta/generation_tagged_payload_lines@1/template.yaml): distinct payload lifetimes |
| Symmetric peer addressing | [Owner reads](delta/peer_read_owned_collective@1/template.yaml) / [multimem capability@2](delta/multimem_peer_address_operation@2/template.yaml): explicit owner map versus exact hardware operation/type/pack |
| Complex schedule with regular payload work | [Bounded descriptor handoff](delta/bounded_schedule_descriptor_handoff@1/template.yaml): scheduler and worker groups, separate routing proof |
| Changing receive slot allocation | [Phase-separated credits](delta/phase_separated_credit_repartition@1/template.yaml): quiescent reset and next-alias peer |
| Sentinel-safe native vector transport | [Rotating poison completion](delta/poison_sentinel_rotating_completion@1/template.yaml): sentinel exclusion, peer skew and output lifetime |
| Quantized expert combine | [Conditional log transport](delta/conditional_log_magnitude_transport@1/template.yaml): explicit lossy codebook and variable mode offsets |
| Multiple engines or queues | [Ordering join](delta/cross_transport_ordering_join@1/template.yaml): join each active completion domain at dependent boundaries |
| Copy-stream latency hiding | [Credit-bounded deferred drain](delta/credit_bounded_deferred_drain@1/template.yaml): delay stays below the producer credit-ring depth |
| Strided network regions | [Byte chunk plans](delta/strided_plan_byte_chunk_transfer@1/template.yaml): exact mixed-radix coordinate reconstruction and final-piece publication |
| Modular packed byte reduction | [Carry isolation](delta/carry_isolated_byte_add@1/template.yaml): independent byte sums without neighboring carry corruption |

Composition: output stripes → local subgroup reduction → network/head reduction → final post-op → local consumer fanout, with a selected credit/tag/direct memory realization. Registered copies may disappear while notification and lifetime obligations remain. Existing DeepEP destination quotient and CUTLASS low-float quantizer are reused at their exact contracts; [source bindings](trace-additional-designs.md) document those comparisons once.

Schema and local checks pass, including1,216 actual PAT source descriptor cases. This is source extraction, with no network launch, timing, formal proof result or rule promotion. Native visibility, progress and approximate numeric policy are explicit premises for future bindings.
