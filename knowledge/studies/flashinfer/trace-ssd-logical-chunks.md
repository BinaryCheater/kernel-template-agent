# SSD: scalar state carry across packed physical chunks

When packed sequences share a physical token block, the logical recurrent
chunks must follow each sequence's subrange rather than the block's full
extent. A causal contraction alone does not isolate histories: clip both
query/key axes, rebase the incoming/outgoing gate frames and keep a separate
state seed/carry for each sequence.

[chunked_decayed_outer_product_state@2](delta/chunked_decayed_outer_product_state@2/template.yaml)
and its [outline](delta/chunked_decayed_outer_product_state@2/proof.yaml) extend
the retained [version1](inputs/reuse-additions/chunked_decayed_outer_product_state@1/template.yaml)
and [outline](inputs/reuse-additions/chunked_decayed_outer_product_state@1/proof.yaml).
Existing64/256 complete-chunk bindings and decay_mode stay addressable. The new
version adds128 and chunk_domain=clipped_physical_subranges, plus explicit
subrange/frame/mask/state-generation conditions. Old prefix/state/numeric
obligations keep their owners; shared_physical_boundaries owns the new case.
No other worker's proposal is edited.

The concrete witness is `flashinfer/mamba/ssd_combined.py` to
`flashinfer/mamba/ssd_kernel.py` in the [retained tree](inputs/sources.json),
using datacenter Blackwell tcgen05,128-token aligned physical backing,
FP16/BF16 IO, FP32 scalar-prefix and carried-state arithmetic. Constructor
`:306-328` of the wrapper rejects targets outside its datacenter Blackwell
major10/11 path; the source's Cake portfolio is a separate implementation menu.

Bind the recurrence's abstract axes as H[dstate,headdim], k=B, q=C,
v=processed_dt*X, alpha=exp(A*processed_dt). Those formulas describe real
values only. The actual processed-dt stages differ:
`triton/kernels/ssd_chunk_state.py:120-148` loads FP32 dt, adds optional bias,
applies optional softplus/clamp, stores dt_out in IO dtype, but computes the
FP32 A*dt cumulative prefix from the pre-store FP32 value. Native insertion
uses that rounded dt_out, while decay uses the FP32 prefix. The declared
numeric binding must retain both stages; recomputing decay from rounded dt
is a different recipe. The state insertion scales B*dt before its MMA, whereas
the abstract v binding absorbs dt into X; that placement also needs its admitted
operand-rounding policy.

| Concrete SSDKernel decision | Rule field / consequence | Evidence |
|---|---|---|
| `:3250-3274,3825-3895`: varlen grid one sequence/head tile; sequence logical prefix selects records(physical_chunk,offset); next record in same physical chunk defines hi | chunk_domain=clipped_physical_subranges, pre_subranges | A physical block can appear multiple times for disjoint sequence ranges. Logical state/output ownership is not the physical block ID |
| `:1374-1427`: per-tile state seed/zero init, FP32 tState, initial IO projection copy | pre_state_generations, numeric binding | Same sequence state is carried across its own logical chunks; new sequence gets its own seed |
| `:1440-1516`: last prefix from hi-1; insertion B*dt*exp(prefix_end-prefix_j); zero outside[lo,hi); subtract prefix_(lo-1) from carried-state decay | pre_segment_frames, pre_masks | State contributions use the sequence's actual local interval, even when lo>0. End correction precedes scaling; beginning correction affects incoming state factor |
| `:1551-1578`: new FP32 state=intra insertion+exp(local_end)*old tState, then IO copy for inter-output consumer | program outgoing-state formula | The next chunk receives inclusive new state; FP32 persistent carry is not replaced by its narrow projection copy |
| `:2326-2370,3945-3996`: full C*B^T MMA, mask both axes to[lo,hi), then causal gate-difference weighting and dt | program local causal term, shared_physical_boundaries | Masking only keys or only causal order leaves another sequence's query rows/support. Upper causal entries become negative infinity before exp |
| `:1987-2013`: subtract prefix_(lo-1) from incoming-output exponent | pre_segment_frames | Previous physical rows' decay cannot scale this sequence's seed |
| `:1674-1695`: final state converts from tState to selected state dtype before store | state boundary/numeric binding | The final checkpoint comes from carried state, not its IO projection copy. Empty logical range preserves its initial state under the declared storage recipe |
| `:2013-2085`: add optional X*D, then multiply optional z*sigmoid(z), then final IO store | separate pointwise post-map | D contribution precedes z gate; neither belongs inside the affine state transition. Seed output_tiling@2 independent point ownership applies after recurrence values are ready |

The four contraction roles are concrete: INTRA1 computes C*B^T, PRE_INTRA
applies causal decay/dt, INTRA2 multiplies by X; INTER1 accumulates the scaled
B/X insertion, PRE_INTER updates carried state, and INTER2 projects the incoming
state with C. Distinct ready/empty pipelines connect those roles. The one-CTA
instruction group and1x1x1 cluster are fixed in this source; a two-CTA branch
being present does not constitute a supported alternative binding.

The [exact check](checks/check_clipped_ssd_chunks.json) compares162 sequence
cases,153 logical subranges and48 shared physical blocks with an independent
serial outer-product recurrence. It includes empty sequences, nonzero distinct
seeds, multiple shared boundaries, both-axis mask checks and a missing-rebase
counterexample. Small4/8-row blocks isolate the combinatorics;258 labeled
native128 coordinates check physical/logical ownership without claiming those
small blocks are native menus. Native dt placement, state IO copies, MMA/TMA
lowering and retirement are separate obligations.

The retained ordered_affine_prefix@1 can describe scalar/diagonal state
coefficients here, but this CuTe witness does not run a full global prefix scan:
it runs local causal contractions and a serial per-sequence chunk-state carry.
Thus the extended chunked outer-product rule supplies the concrete computation
organization; it does not claim the matrix-affine CP fixup pipeline used by GDN.
This source supplies the inter-chunk propagation missing from the earlier
TileLang partial chunk-state/output witnesses, including clipped packed domains.

The underlying source scheduler, codec/checkpoint replay, Cake route and
training families still require their distinct bindings in the
[recurrent inventory](recurrent-inventory.md); this trace disposes the CuTe SSD
computation/carry and shared physical boundary mechanism.
