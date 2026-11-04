# NVIDIA design inventory

Reconciled retained archive against `src/device/generate.py`, `symmetric/generate.py`, public device primitive headers and contrib device families. [Concrete bindings and source locations](trace-additional-designs.md), [original ring/FIFO witness](trace.md). CUDA C++/PTX, source-defined architecture/toolchain gates; no selected runtime device or performance measurements.

| Distinct design | Disposition |
|---|---|
| Ring AllReduce, reduce-scatter, allgather | ring_reduce_scatter_allgather@1 with phase-specific consumers and ring-to-user-rank map |
| Root-cut broadcast/reduce | Source-bound root/terminal predicates around same streamed primitive; post-op once at completed root |
| Tree up/down and split roles | subtree_reduce_root_broadcast@1; constructor protocol participation and disjoint barrier groups |
| CollNet direct/chain, NVLS/NVLS-tree | head_partitioned_hierarchical_collective@1; registered layout permutation, local/network contract and unique finalization |
| NVLS/CollNet AG/RS rail layouts | Dense-to-user rank and byte/element interval mapping, plus hierarchy phase specialization |
| SIMPLE FIFO / registered direct pointers | credit_gated_circular_transfer@1 + acknowledged_direct_buffer_exchange@1; descriptor acknowledgment differs from payload retirement |
| LL / LL128 | generation_tagged_payload_lines@1 conditional native visibility, mapping, flag lanes and rollover cleanup |
| PAT single/multi-RPN RS and AG | Source-bound dimension/phase schedule; bounded_schedule_descriptor_handoff@1; separate accSize seed/append ownership, NVLS local phase and skipped-step retirement |
| Symmetric owner-read LD/ST and TMA | peer_read_owned_collective@1; paired ready/done source lifetime and owner coordinate partition |
| Symmetric small-message LL/LLMC | Gather signed-generation packs then local reduction; epoch banks, complete collective retirement and finite-tag cleanup |
| Multimem payload and credit operations | multimem_peer_address_operation@2 capability revision, supported operation/type/pack records; source barriers remain required |
| Symmetric GIN rail ring / all-to-all RS | Head/rail hierarchy, separate outbox request completion and inbox receive credits; phase_separated_credit_repartition@1 |
| Typed/vector reduction, pre/post and one rank | carry_isolated_byte_add@1; arithmetic cast/round/wrap policy and pack alignment; original-source pre-op / final-only post-op |
| Batch loader/dispatch, profiler | Shared complete-record staging with bitset rank and separate parameter/global address spaces; stable workStorage through readers |
| Network iovec unpack plugin | Per-warp metadata staging, alignment-selected pack copies and byte tail; NIC/provider readiness external |
| contrib custom allreduce/alltoall / UBX | Owner fan-in versus duplicate per-rank reduction, push/pull expert paths, poison_sentinel_rotating_completion@1, FP8 metadata and RMS postwork |
| contrib nccl_ep low-latency | DeepEP destination_quotient_transfer@1; internal128-channel E4M3 versus external byte forwarding, conditional_log_magnitude_transport@1, metadata-complete weighted combine |
| contrib nccl_ep high-throughput | Routing scan, node quotient, N2N/G2S/S2G independent worker stages, marked-last contributor reduction, local expansion/reduction and complete bundle tx |
| contrib nccl_m2n | strided_plan_byte_chunk_transfer@1; direct/ring signal counts, local follower fanout and transpose staging |
| contrib PACE | Ring/mixed RS/AG, virtual tensor layouts, local P2P groups; credit_bounded_deferred_drain@1 for copy-stream/graph pipeline |
| contrib NIIN | LSA versus GIN RMA, TMA staging, peer/system/native/proxy atomics, cross_transport_ordering_join@1; unsupported API paths explicit |
| contrib Rust shim / examples and benchmarks | ABI forwarding and instances of public primitives; no distinct production schedule |

No material family is closed from directory names alone. Unproved native payload/tag atomicity, GPU/network progress, host-proxy atomic linearization, reduction numerical policy and architecture capabilities remain explicit binding/proof obligations, not missing source families. PAT CPU replay checks descriptor legality/termination, not full distributed value correctness.
