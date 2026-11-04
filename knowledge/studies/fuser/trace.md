# Concrete reduction and variant bindings

Caller: generated NVIDIA CUDA invokes blockReduce or gridReduce with axis masks, identity, read/write predicates, block dimensions and scratch. The runtime helpers are device bodies rather than standalone launchers. No emitted application kernel is claimed.

| Source and binding | Rule / field | Evidence and status |
|---|---|---|
| runtime/block_reduction.cu:34-65; block_dim=(4,3,1), reduce Y only | reduction_partition@1 assignment=strided_threads; physical_storage@1 level=smem | Four independent outputs, three values each; read_pred=false supplies identity, not an uninitialized value |
| block_reduction.cu:119-174; reduction_size=3, np2=2 | reduction_partition@1 exact_once | Fold y=2 into y=0, then merge y=1; peer_stride=4 preserves x. Non-power-of-two path is distinct from padding all lanes |
| grid_reduction.cu:218-272; reduce block X, grid=(3,2,1); prior thread X reduced | reduction_partition@1 contiguous_parallel; physical_storage@1 gmem | Two independent segments. Work address is entrance/segment base + block-x * surviving-thread-count + surviving-thread index |
| grid_reduction.cu:273-323; nonpersistent | designated_segment_completion@1 mode=one_shot | Designated last coordinate waits, not the last arriving block. Each entrance uses distinct scratch and semaphore. Other blocks may exit |
| grid_sync.cu:17-87; semaphore initially zero | designated_segment_completion@1 publication/progress | Fence, CTA barrier, one atomic arrival per CTA; high bit flips after all arrivals. Designated block adds 2^63-(N-1), other blocks add one. Source scheduling assumption is retained as an unresolved progress obligation |
| grid_reduction.cu:273-323; persistent | designated_segment_completion@1 mode=reusable | All blocks wait before cleanup; a second grid join protects work-buffer reuse. All waiting participants must fit and make progress |
| runtime/welford.cu:49-68 | centered_moment_merge@1 count_mode=explicit (provisional TE rule) | Carries mean,M2,N; b_N=0 is identity. Delta=b_mean-a_mean; correction delta²*a_N*b_N/(a_N+b_N). Unequal counts must not use equal_inferred |
| runtime/scan.cu:39-129 | CCCL BlockScan boundary | Inclusive, ITEMS_PER_THREAD blocked items; every non-unit thread axis participates; init_value is unused. Do not reinterpret this as exclusive scan |
| runtime/argsort.cu:46-95 | Local instance, full radix ordering | Value payload is thread_id*ITEMS_PER_THREAD+i; ascending/descending changes order, not ownership. Full-block participation is a static requirement |
| csrc/scheduler/normalization_inner.cpp:30-61 | Policy versus legality | TMA route checks contiguous inner dimension, 128-bit vector width, persistent-buffer register budget and at least 128 post-vectorization threads. These are implementation menu guards; no measured preference inferred |

Composition: reduction_partition@1 produces per-CTA values → physical_storage@1 holds separate global partials → designated_segment_completion@1 joins publications → gridReduceLastBlock folds them using blockReduce. Completion publishes values; it does not select numerical merge order.

Closest pending rule: epoch_tagged_completion@1 fixes an initializer/128-wide epoch scheme and 1..64 arrivals. Fuser has a high-bit phase and designated coordinate, so binding that rule would silently change protocol and progress premises. first_writer_atomic_combine@1 combines into one numeric accumulator; Fuser publishes separate partials. Neither is a direct reuse.
