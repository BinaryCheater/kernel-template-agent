# B12x streaming compression and MLA relatives

Source paths are relative to the retained [tree](inputs/sources.json).
This pass opens the B12x attention family under
`flashinfer/experimental/b12x/attention/`. Its preparation/API files select
retained CuTe/Triton kernels; Python `is_supported()` and compiled menus are
source applicability metadata, not native runtime evidence.

## Stream pairs need a persistent boundary protocol

Retrieval cue: adjacent even/odd compression receives packed independently
advancing requests, and call boundaries can split any pair. New
[stream_pair_boundary_state@1](delta/stream_pair_boundary_state@1/template.yaml)
and its [proof](delta/stream_pair_boundary_state@1/proof.yaml) keep one tagged
unmatched even token per stable stream, independently of packed row ordering.

Route: `mla_compress/api.py` -> prepared `_impl.py:343-389` ->
`_cute.py:_Compress`. Ratio1 only normalizes individual BF16 rows. Ratio2 uses
FP32 value/gate payloads and the following protocol.

| Source decision | Candidate field / obligation | Binding |
|---|---|---|
| `_cute.py:110-138` | `pre_segments`, `tag_identity` | Binary upper-bound search locates the request containing each token, skipping empty ranges. Absolute position=start_position+token-first; request-row order is not stream identity. |
| Same file :140-145 | `chunk_invariance` | Only odd positions emit. An odd first row requires pending_position[state_id]==position-1; an interior odd row reads its preceding even row in the same request. A stale position tag does not match a discontinuous pair. |
| Same file :40-89 | `execution=parallel_read_then_commit`, `read_commit_order` | Main output launch reads old carry; a second ordered launch commits request tails. A CTA-local barrier cannot substitute for this all-reader boundary. |
| Same file :203-237 | Tail update | A nonempty valid request ending even stores that last value/gate and absolute tag. Ending odd invalidates the tag. Empty/inactive requests do not update it. Payload/tag publication relies on ordered completion before the next invocation, not a concurrent atomic record store. |
| Same file :178-194 | `processor_and_emission` | Output stays at the odd row, with emitted flag and that row's caller destination slot; nonemitting rows are zero with invalid emission metadata. |
| `_impl.py:96-173` | Buffer/ABI boundary | Capacity-shaped caller-owned tensors and disjoint mutation storage are bound ahead of launch. Unique live state IDs and invalidation on state-slot reuse remain semantic caller premises; a position tag alone cannot distinguish two streams reusing the same ID/position. |

The serial-per-request alternative in the candidate is an inferred organization;
the source native representative is the two-launch parallel form. Existing
adjacent_boundary_reuse@1 shares sampled finite-difference endpoints, with no
persistent stream identity or cross-launch overwrite protocol. It cannot be
bound just because both designs mention an adjacent value.

### Pair arithmetic is an explicit checkpointed operator

`_cute.py:155-193` computes an independent two-way softmax of gate0[d],gate1[d]
**per channel**, pools the two FP32 values, rounds the entire pooled vector to
BF16, then computes ordinary weighted FP32 RMSNorm with epsilon1e-20 and final
BF16 output. Source width is512. It is not one scalar mixing coefficient and
not a fused unrounded pooling/RMSNorm identity. `reference.py:14-28` supplies
an allocation-using math oracle only; it is not the runtime path.

The [logical check](checks/check_stream_pair_state.json) covers2040 chunking
cases, nonzero starting parity, absent/present carry, empty requests,
discontinuous tags and state-slot reuse. A call starting at odd1 and ending
even2 must read pending even0 before committing even2; early commit loses the
odd1 output. A512-channel numeric witness distinguishes BF16-before-RMSNorm
from final-cast-only normalization. Native block reduction, fast exp/rsqrt,
CuTe code and stream dependencies remain realization obligations.

## B12x MLA relatives: concrete reuse and distinct numeric boundaries

| Representative / source | Existing rule or explicit gap | Disposition |
|---|---|---|
| `compressed_sparse_mla/_metadata.py:18-52` | indexed_gather@1 map identity; stable_sparse_support_compaction@1 only if separately selected | Logical row IDs become table[row,logical//page_size]*page_size+logical%page_size. Prefix length, negative IDs, physical-page range and int32 slot bound guard the map. This source preserves positions/holes in the output rather than silently packing them. |
| `dense_mla/planner.py:55-89`; `_shared/mla/compressed_config.py:22-76` | reduction_partition@1 and runtime_partition_capacity@1 address/control premises | Capacity-static planners choose contiguous candidate chunks from wave/capacity formulas. Decode-row capacity remains stable for captured smaller replays. Chunk menus12/64/1024 and the three-wave search are policy, not a performance result or new partition algebra. |
| `dense_mla/_math.py:263-379` | online_rescaled_reduction@2 running_max,gain1 | Warp-local score maxima/sums combine into a CTA frame; old PV registers and mass rescale before new common-frame contributions. Exact QK component maps, FP8 descales and per-owner value coordinates remain native bindings. |
| Same file :710-780 | split_combine_kernel@1 representation binding | Partial output is normalized by local row sum; FP8 V descale applies in this output stage. Pair this with the source LSE base/empty sentinel before choosing a later merge, rather than interpreting partial O as a raw numerator. |
| Same file :383-394 | Two-code probability residual gap | `_quantized_probability` returns high=E4M3(bounded P), low=E4M3(bounded P-decode(high)). PV uses the two-code representation. This differs from tile_anchor_probability_encoding@1 and from a single gained P code. A dedicated two-contraction numeric binding remains open. |
| `_shared/mla/kv_cache.py:338-386` | groupwise low-float / hierarchical decoder concepts, explicit cast-order specialization | DSV4.1 floors group amax, computes round-to-nearest division by6, casts scale, decodes that stored scale, then divides values by it with RN arithmetic before E2M1 packing. The sibling path uses approximate reciprocal; moving either division into reciprocal multiplication changes exact midpoint behavior. Do not claim bitwise reuse from matching record widths. |
| Same file header and per_token_scale branch | hierarchical_low_float_scaling@1 row-domain concept; additional zero/range premise | Optional non-DSV4.1 per-token outer scale occupies bytes292..296, with decoder q*s_group*s_token. Static versus inline-scale mode is compile identity; zero padding in old records is not valid scale metadata for the new mode. TE's positive-scale recipe needs the actual source floor/zero policy bound before exact program reuse. |

Cache record families remain distinct:528-byte FP8 latent plus four FP32 scales;
304-byte NVFP4 latent with inline token-scale field;368-byte NVFP4 latent plus
E4M3 RoPE and FP32 RoPE scale. The source's DSV4.1 prepared writer has its own
swa/indexed geometry. Record width alone is never the decoder identity.
First-occurrence duplicate ownership from the earlier Cake writer is not
inferred for B12x append unless its native election map is matched.

## Remaining B12x scope

This pass establishes stream compression and concrete MLA reuse boundaries.
Paged/varlen, sparse/prefill shared MLA, QSA draft selection and DSA selectors
remain in the full library inventory; their target-specific tile/pipeline
choices need representative dispositions. The two-code probability residual,
common orthogonal preconditioning and ExpCast codec remain named extraction
work. Continue those design differences, then reconcile attention/training
before advancing through GEMM and the remaining families.

The two-code probability gap is resolved in the [dual-P/selection follow-up](trace-b12x-residual-p-and-selection.md); remaining maps/protocols retain their stated scope.
