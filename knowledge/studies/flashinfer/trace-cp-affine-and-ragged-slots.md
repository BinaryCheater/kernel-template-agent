# Parallel segment summaries, ordered fixup and ragged storage

When a long affine recurrence has too few sequence/head chains, summarize
independent contiguous pieces, propagate their entry states through a shorter
ordered fixup, then compute local outputs in parallel. The retained source's CP
means intra-GPU context parallelism; it does not provide a distributed collective.
Its ragged workspace also avoids a second prefix scan of derived chunk counts.

This pass reuses and extends the provisional
[affine_piece_prefix_state@1](../sglang/delta/affine_piece_prefix_state@1/template.yaml).
The exact input pair is [retained locally](inputs/reuse-additions/affine_piece_prefix_state@1/template.yaml)
with its [outline](inputs/reuse-additions/affine_piece_prefix_state@1/proof.yaml),
recorded in [reuse additions](inputs/reuse-additions.json). The new
[affine_piece_prefix_state@2](delta/affine_piece_prefix_state@2/template.yaml) and
[outline](delta/affine_piece_prefix_state@2/proof.yaml) add `prefix_supply`:
per-piece prefix replay or one shared ordered fixup. Version1 is unchanged in
its owner's directory. `piece_chunks` keeps its extent meaning; version2 allows
runtime binding, with a constant specialization retaining version1's extent
choices. Existing pre_affine/pre_numeric and composition/publication/numerical
obligations retain their owners; new conditions own inclusive-boundary storage,
exclusive-entry reads and boundary lifetime.

For V-major delta state, one token is right affine:
H_next=H*A+B, A=alpha*(I-beta*k*k^T), B=beta*v*k^T.
An earlier map(M0,N0) followed by(M1,N1) composes as
(M0*M1,N0*M1+N1). It is associative in exact arithmetic and not commutative.
This matrix transition cannot bind
[ordered_affine_prefix@1](../mamba/delta/ordered_affine_prefix@1/template.yaml),
whose pre0 and scope explicitly require coordinatewise/diagonal state and
exclude general matrices.

Paths below are relative to [retained FlashInfer](inputs/sources.json), mainly
`gdn_kernels/delta_rule_dsl/delta_rule_cp_sm90.py` on SM90a, D128.

| Source decision | Exact rule binding / boundary | Evidence |
|---|---|---|
| `:4684-4709,4887-4945`: T precompute, local M/N, fixup, local prefill in ordered launches | `prefix_supply=shared_fixup`; version1 pre_affine/pre_numeric plus new boundary lifetime | Completed artifacts publish between stages; independent M/N construction does not assume the incoming state |
| `:175-225`: masked strictly lower KKT/beta, inverse, fused negative-beta T store | chunked_delta_feedback@2 equations with source sign/orientation conversion | This T is a negative-beta representation, not the positive T array from another GDN path. Bind the actual producer/consumer pair rather than reusing bytes |
| `:892-1002`: Xt, transfer update M<-gamma*(M+M*K^T*X), state offset update N<-gamma*N+Y*X | version1 program, `L=M,c=N`; `piece_chunks=cp_chunk_len/64` | First transfer is identity, first offset zero; piece-local64-token steps fold in chronological order. X includes the signed weighted solve; Y includes the negative value term |
| `:1749-1752,2846`: FP32 M,N and fixed-state storage | summary/fixed precision premise | Three128x128 FP32 matrices per allocated piece/head across stages, not BF16 summaries inherited from the SGLang witness |
| `:2390-2518`: SIMT fixup initializes seed or skips first zero-seed multiply, applies N+state*M, stores every inclusive boundary | shared ordered fixup | Sequence/head/row-shard ownership retains full key reduction; each chunk reads the previous completed state. Shared-memory publication separates updates |
| `:1968-2085`: HMMA fixup converts state/transfer operands to TF32 and accumulates into N; use_3xtf32=False in constructor | numeric backend boundary | Actual default is single TF32 product, not three-product correction merely because a disabled branch exists. SIMT FP32 and HMMA TF32 are different recipes |
| `:2862-2877`: heads<=8 selects simt_row4,<=16 simt_row8, otherwise hmma | compiled menu versus tuning | Row tile selection changes parallelism/register work. No numerical interchangeability follows from dispatch |
| `:3945-3965`: local piece0 uses initial seed; later piece r reads fixed slot r-1; only last valid piece stores final state | new exclusive_boundary_map | Fixup writes inclusive after-piece states, but replay consumes exclusive entry states. A global slot>0 does not imply a previous piece within this sequence |
| `varlen_helper.py:257-320`: runtime CP length from parallel-wave/balanced-chain policy,64 aligned | runtime piece_chunks | Piece length trades parallel local work, summary storage and fixup length. Heuristic ratios/device thresholds are cost policy; legality comes from supported64-aligned lengths and capacity |

The [focused check](checks/check_cp_affine_and_ragged_slots.json) compares32
matrix-affine final states and88 exclusive piece entries against an independent
serial delta recurrence. It explicitly compares both prefix supplies, uses
nonzero/zero seeds, short final pieces, and a noncommuting-order counterexample.
It does not establish native summary/TF32 error or synchronization.

## Bound-derived ragged workspace

[prefix_bounded_ragged_slots@1](delta/prefix_bounded_ragged_slots@1/template.yaml)
and [outline](delta/prefix_bounded_ragged_slots@1/proof.yaml) capture an additional
storage choice not supplied by the affine summary rule.

For N sequences, total T, existing token prefix S_i and chunk length C, define
B(n,t,C)=m+floor((t-m)/C), m=min(n,t).
Sequence i's live chunk records start at B(i,S_i,C), have ceil(L_i/C) entries,
and fit inside the host-known capacity B(N,T,C). The next sequence's bound
start is at least this start plus its live count. Gaps may occur; consumers
visit only actual local chunks, so gaps need not hold initialized states.

`varlen_helper.py:133-147,321-329,361-381` supplies matching host capacity,
device bound, segment/local address and valid-length functions. Source M/N,
fixup and prefill all use them. This reuses existing token-prefix metadata;
it avoids a new scan/readback of ceil(length/C), not the original sequence
metadata itself. Dense logical work enumeration remains a separate mapping.

The same check enumerates10878 small/native-aligned partitions, including7509
with unread gaps. It validates exact disjoint live coordinates and capacity,
and shows two failures: ceil(total/C) can underallocate many short sequences;
and global previous-slot indexing can select a gap before a sequence's first
chunk. The latter must use its original seed, guarded by local chunk index.

[segmented_aligned_storage@1](../cutlass/delta/segmented_aligned_storage@1/template.yaml)
was read from its retained proposal without resuming CUTLASS. Its program
prefix-sums separately rounded segment extents; this source instead computes a
conservative prefix bound directly from original tokens and segment count.
The resulting starts/capacity can differ, and unread gaps are not required to
be neutral payload. Likewise
[runtime_partition_capacity@1](delta/runtime_partition_capacity@1/template.yaml)
separates useful counts from capacity but does not supply this ragged bound map.

This disposes the CP summary/fixup/output organization and its conservative
storage map. Target-specific pipelines can reuse these invariants only with
matching summary precision, coordinate/readiness and supported instruction
bindings. State-locality/page generations, diagonal SSD propagation, codecs
and training remain distinct representatives in the [recurrent inventory](recurrent-inventory.md).
