# SM120 convolution input coordinates and completion

When a convolution's input data and scale metadata use different transfer
engines, require both to select the same logical input pixel, channel group and
filter tap before joining their readiness. Retrieve
[implicit_patch_contraction@1](delta/implicit_patch_contraction@1/template.yaml)
for the spatial map and
[byte_arrival_completion_join@1](delta/byte_arrival_completion_join@1/template.yaml)
for the two completion counters. Neither rule supplies an arbitrary scale layout
or a compressed coordinate representation: those need explicit bindings.

This continues the [frontend slice](trace-geforce-conv-frontend.md). Source paths
below are under `examples/python/CuTeDSL/cute/blackwell_geforce/kernel/conv/` in
the [retained tree](inputs/sources.json). `scaled` means
`dense_blockscaled_implicit_gemm_fprop.py`; `dense` means its unscaled companion.
Both device entries remain active: this trace extracts input coordinate arithmetic
and completion incidence, not complete output/scheduler closure.

## Input mechanism decisions

| Source | Observed choice and concrete binding | Existing field or remaining obligation |
|---|---|---|
| Scaled `__call__`:434–556 | A view((W,H,D,N),C), B view(K,(C,S,R,T)), D view((Q,P,Z,N),K). Runtime filter extents retain all four B reduction modes, even when a tap extent is1. Nested B tiler tiles channels while retaining S/R/T in the rest modes. | `implicit_patch_contraction@1.program` fixes the logical map; `pre_backend` still requires native layout/transfer realization. This is not permission to flatten padded channel tiles across filter positions. |
| Scaled `kernel`:1750–1857, `_make_b_k_coord`:773–786 | Within each output tile, iterate channel tile, S, R, T. B is explicitly indexed by this four-tuple, A advances the corresponding hierarchical coordinate iterator. SFA independently advances identical counters. | Source input traversal binding. Each position has ceil(C/tileK) data tiles, with zero-filled channel tails. Logical reduction order and inserted neutral terms still require `matrix_arithmetic_admitted`; no floating regrouping equivalence follows merely from coordinate agreement. |
| Scaled SFA full helper:804–952 | One of128 lanes owns one M row; decode Q/P/Z/batch, then spatial input=(output*stride−lower_pad+tap*dilation). Predicate includes row and all spatial bounds. Invalid spatial axes and batch are clamped before forming an address; actual copy is predicated off. | Exact old `program` and `patch_identity` map, now applied to the scale tensor's input-pixel row. The scalar address remains in range even for masked rows. Native false-predicate encoding and copied SF values remain required. |
| Scaled NVFP4 branch:1540–1663 and row helper:955–1049 | Hoist M decomposition and input base once per output tile; recompute the row address and predicate only when the channel counter carries to a new filter position. H/W bases occupy signed16 halves of one Int32 register. | Conditional implementation optimization. Old patch mapping describes the result but has no field choosing hoisting or bit packing. No reused field is claimed to supply that extra choice. Exact equivalence requires both bases in[-32768,32767] and representable row arithmetic. Larger bases need a wider representation or a guard. |
| Scaled SFA helpers:858–870,913–948,988–1045 | SF K span=max(tileK,4*vector); transfer whole four-byte SF chunks. Base scale group=floor(channel_tile/ratio)*(SFspan/vector), ratio=SFspan/tileK. If a whole chunk lies beyond C, re-read that pixel's first chunk. | Explicit source-tail policy. Whole-atom C alignment ensures a chunk never partially straddles a real row. Native values must be finite where zero-filled data is multiplied; neither a legal address nor zero data establishes 0×scale=0 for every encoding. |
| Scaled SFB producer:1746–1814 | SF N coordinate=floor(data_N_tile/n_tiles_per_SF_N_tile). Flatten filter position as s+S*(r+R*t); SF K tile=position*ceil(C/SFspan)+floor(channel_tile/ratio). | Per-filter-position padding and scale transfer cadence are distinct from A/B K width. N96 is legal only for one tile, so its nondividing128-row SF chunk is not generalized to multiple tiles. Native N/chunk lane maps are still open. |
| Scaled consumer:2059–2076 and subsequent loads | `sf_base=((local_stage_count % channel_tiles_per_position) % ratio)*MMA_K_blocks`; read the live part of the SF chunk. The whole chunk is loaded **again for each A/B stage**, not retained once for several stages. | Existing grouping rules do not infer this source schedule. Tested logical code/scale pairing; native filtered layout and LDSM composition remain unverified. Connected NVFP4 has ratio1, MXFP8 K64/vector32 has ratio2. |
| Scaled setup:219–245 | The shared MMA helper is called with actual A/B types. Mixed-mode register shifting exists in the body, but the host menu supplies equal-type A/B. | No mixed-dtype convolution binding is connected. A nearby comment still says only FP4 is allowed; actual `can_implement` also accepts MXFP8 E4M3. The executable predicate takes precedence over that stale comment. |

The hardest conditional reuse is the data/scale spatial mapping. The old patch
rule's `program` supplies **lower-side** padding and tap orientation; a generic
scale-group label cannot supply those decisions. The source makes its two engines
share runtime geometry, but the software address still needs independent checking.

## Completion incidence and partial schedule binding

Scaled `kernel`:1159–1200 counts only A/B/SFB TMA bytes in `tma_copy_bytes` and
initializes129 full-barrier arrivals:128 SFA lanes plus one TMA registration owner.
The TMA warp waits for empty and calls `sync_object_full.arrive` before issuing
the three transfers. `pipeline/helpers.py:356–393,458–469` dispatches this to an
elected `mbarrier_arrive_and_expect_tx`; this registers bytes and consumes the
reserved arrival. Every SFA lane waits for that same empty generation, issues
its cp.async chunks and calls `arrive_cp_async_mbarrier` after commit. The helper's
`cp_async_mbarrier_arrive_noinc` supplies copy-completion incidence. No artificial
two-group delay is inserted here; both producer branches drain at exit.

This binds the old join rule's `copy_arrivals=128`, transaction-byte count,
reserved owner and correct generation. Source incidence is established; native
cp.async/TMA ordering, scale encoding, arrival visibility and progress remain
backend premises. The SM100 source's delayed-arrival schedule is not imposed on
this immediate-arrival source.

Both convolution sources use CLC work discovery and let both math groups see
every returned tile. A local parity selects one owner; the other group advances
input and optional bias/residual cursors without releasing those stages. This is
a candidate binding of `interleaved_consumer_streams@1`: ownership is by local
stream ordinal rather than cancelled-grid tile number. Unlike the earlier
standalone GEMM declaration, both convolution order arrays explicitly reserve
four Int64 slots for depth2/length2. Dense CLC readers count320 lanes; scaled
counts448 without residual and480 with residual. The unchanged failed-CLC-query
wrapper gap remains; scheduler/exit protocol and all side-stream lifetimes are
still to be replayed before this conditional binding is closed.

## Recovery checks and a representation boundary

[Input check](checks/geforce_conv_inputs.py) executes the retained SFA scalar prefix,
NVFP4 hoisted base/carry expressions, actual TMA filter/channel counter update,
SFB index and consumer SF-base expressions. It tests16 asymmetric spatial
geometries/16384 row-tap cases,100 signed packing cases and26 connected format,
channel/tile/tap domains/38272 scale-element selections. Distinct coordinates
separate lower from upper padding. Every live channel selects its own scale group;
whole padded chunks re-read an in-bounds chunk. [Result](checks/geforce-conv-inputs-result.json)
records these finite domains. Signed packing models an Int32 register explicitly;
Python's unlimited-width left shift would hide its truncation boundary.

A source-menu-accepted NVFP4 witness is input N1/C64/D1/H32769/W2, filter
K128/T1/R1/S2, padding0,stride/dilation1,tile128×64×64. Output Q1 makes row32768
start at H32768/W0. The first tap uses that valid H. After the first channel tile,
the S counter carries to1 and the source rebuilds packed H as **−32768**; the
correct second-tap coordinate is H32768/W1. The rebuilt predicate is false even
though the correct coordinate is valid. Frontend acceptance and the Int32
coordinate discrepancy are checked; no native numerical failure is claimed.
Restricting both initial H/W bases to signed16 range makes the tested packing
identity hold. The study records this as an implementation boundary, not a new
convolution algorithm or a silently broadened old rule.

The earlier C192/MXFP8 span example tests helper arithmetic only: its C fails
the actual128-channel scale-atom alignment gate. Here connected MXFP8 domains use
C128/256/384 with tileK64; NVFP4 also includes C64/192 and tileK128 tails. Per-position
padding remains a real NVFP4 obligation (C192 rounds to256 at tileK128).

Next: finish both math/epilogue paths, CLC and side-ring tail protocols, host scale
preparation and reference arithmetic. No new candidate is added in this slice.

The [output companion](trace-geforce-conv-outputs.md) now covers math/epilogue
expression order, separate side-stream cursors and output source retirement.
Host scale preparation and layout/resource reconciliation remain active.
