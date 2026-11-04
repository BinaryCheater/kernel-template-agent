# Current NVIDIA design inventory

Reconciled the retained source tree, public operation modules and build/dispatch registrations. Utility/cache/DSL helpers are assigned to their consuming kernel families, rather than counted as independent operators. [Complete source listing](inputs/source-files.txt); [concrete design ledger](designs.md) supplies mechanisms, exact rule fields, choices, constraints and representative source locations. The original inventory is preserved in inventory-initial.md.

| Read representative / design group | Disposition and binding |
|---|---|
| Sequence-contiguous fwd.cu:85–135 | CTA owns one batch/channel, threads own kNElts consecutive tokens. Chunk length=threads*kNElts. t,i output consumes x[kNElts+i-(W-w-1)] with w0 oldest. output_tiling@2 tile_m=kNElts,tile_n=1,executor=thread,index_map=custom inside each batch/channel CTA (chunk grouping is a separate outer schedule); reduction_partition@1 group_size=W,num_groups1,contiguous_sequential for stencil taps. |
| Channel-contiguous fwd.cu:206–340 | Shared x_smem[W-1+Ltile][Ctile+kNElts]. Load lane tid→l_idx=tid/threadsPerC,c_idx=tid%threadsPerC, vector channels c_idx*kNElts. Compute retile: row_idx=tid%Ctile,col_idx=tid/Ctile, outputs col_idx*kLPerThread+i, reading shared time col_idx*kLPerThread+i+w. |
| Initial/final state fwd.cu:253–275 | First chunk negative-time halo uses initial_states; later chunks read original x. Last chunk emits history index L-W+1+j by shared offset L+j-chunk_start. |
| Segment masking fwd.cu:304–334 | Output id<0→0; tap contributes iff source and output sequence ids match. Initial history inherits first sequence id. |
| Streaming update update.cu:61–110 | causal_window_state@1 widthW,history_capacityH. Ring reads(W-1) preceding positions, writes current at p modH and advances local update_idx; cache_seqlens remains caller-owned. Linear path shifts retained history and appends current values while register window provides old samples. |
| Sequence-contiguous backward bwd.cu:89–209 | Reverse token traversal exchanges future dpreact halo rather than past x halo. dx_t=sum_w weight[w]*dpreact_(t+W-w-1); weight partial sums x_t*dpreact_(t+W-w-1). |
| Channel-contiguous backward bwd.cu:461–545 | dx_t=sum_w weight[W-1-w]*dpreact_(t+w), same sequence id required. dweight_w sums x_(t+w-W+1)*dpreact_t. reduction_partition@1 partial over chunk/time then batch; dbias sums dpreact. |
| State gradients bwd.cu:537–570 | history_suffix_pullback@1 final-history df[j] routes into dx for last current tokens, and dh when L<W-1. Initial history h_i additionally receives sum_w weight[W-1-w]*dpreact_(i+w-(W-1)), only valid output times. |
| Determinism causal_conv1d.cpp:394–459 | Separate FP32 workspace [batch,channel,width] or [batch,Lchunk,channel,width]; kernel owns each partial and host tensor sum finalizes batches/chunks, then adds to requested destination. Nondeterministic variant atomicAdds directly. |

Additional numerical/representation branches described outside the tables are part of the same ledger: Short-invocation initial/final-history gradients and the full layout/width/dtype dispatch boundary.

Coverage: materially distinct designs have concrete dispositions; remaining candidate obligations concern native lowering, memory ordering and numeric assurance. No major unread family is used as a completion exception.
