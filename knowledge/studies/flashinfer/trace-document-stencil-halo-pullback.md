# Route stencil history and transpose edges by document position

When a packed causal stencil spans rank fragments, route history by absolute
document token, not just the nearest predecessor fragment. For backward,
return FP32 boundary-output preactivation adjoints to the input owners, then
fold the appropriate taps into their FP32 input gradients before final casting.
Operands with an identical row plan may share exchanges by packing channels.

[document_stencil_halo_pullback@1](delta/document_stencil_halo_pullback@1/template.yaml)
and its [outline](delta/document_stencil_halo_pullback@1/proof.yaml) expose
per-operand versus channel-packed exchanges. The [retained source](inputs/sources.json)
is the public `cake_causal_conv1d_det_train_backend/` and its canonical KCP
facade, not the separate experimental non-deterministic package. Source target
is SM100/103/107, width4 depthwise convolution plus SiLU, BF16 activations/dy/dx,
FP32 taps, no bias and channels divisible by8.

For taps w0..w3, output t reads x[t-3]..x[t]. A fragment [a,b)
needs history tokens a-3..a-1; its input-gradient tail needs future dpre
at output tokens b..b+2. Each position resolves its actual document owner.
Three one-token predecessors can therefore supply a three-row halo; neither
all history nor all reverse messages necessarily belong to one neighbor rank.

| Concrete decision | Rule field / binding | Evidence |
|---|---|---|
| Absolute token owner lookup, ignoring empty fragments | pre_partition / pre_routes | GDN KCP `cake_kcp_plan.py:1310-1356` groups/sorts by document offset and finds each history/future token owner. The same ContextDetKcp halo protocol is used by the conv backend |
| Destination-major sends, source-major receives, row splits | pre_slots | `_halo_plan:1360-1428` builds hist_index/hist_row_off and reverse tail/head tables. Sentinels before/after the document mean no row read |
| One head dpre may serve several predecessor fragments | pre_slots / transpose_edges | head_row_off[n_frag,3,3] has up to3 consumers. Each consumer receives an identical FP32 dpre but applies it to different owned input edges, so this is required replication rather than duplicate dx accumulation |
| Forward gather and wait before history-aware local stencil | pre_lifetime | Conv `kcp.py:_fwd_gather/_fwd_launch/_kcp_forward_trim`; native fwd675940a7a4fc0dba4b53 reads history via sentinel-checked rows at `csrc/*_kernel.cu:296-478` |
| Boundary head kernel recomputes FP32 preactivation and SiLU derivative | pre_stencil | Native head17cd42fb05079bbbf840 `:221-240`: tap0 product then tap1/2/3 FMA chain, approximate exp2/reciprocal sigmoid, dy*sigmoid*(1+pre*(1-sigmoid)) |
| Return raw FP32 dpre before source-owner tap fold | pre_backward | Conv `kcp.py:1098-1245` launches head kernel, exchanges dpre_head, waits and launches main. Main5288d132a5cc927ae729 at `:540-650` loads tail rows and applies weighted FMAs into pending FP32 dx |
| One final BF16 gradient cast after local and remote edges | pre_backward | The native pending ring incorporates future dpre before emitting its owned dx. Returning narrow dx contributions for a later sum would change this stage contract |
| Pack Q/K/V channel columns under one shared row plan | operand_exchange=channel_packed | `kcp.py:_kcp_forward_multi` and `:1828-1890` concatenate row columns, exchange with total channel width and unpack exact channel prefixes. Single-operand calls bind per_operand |
| Empty ranks still exchange and emit zero tap gradients | pre_backend | `:1098-1245,1447-1500`: exchange/wait precedes empty return; dw_out is zeroed. Rank-local emptiness does not prove no global communication |
| Retained history can avoid checkpoint forward exchange replay | pre_lifetime | Facade recompute_handle and `_handle_hists` retain the original histories for the same layer/microstep/context. Reverse halo still runs; this is not a no-communication backward |

## Numeric boundary and distinct backward scheduling

KCP backward differentiates the FP32 preactivation. The separate zero-history
training backend differentiates a BF16-rounded preactivation; identical forward
output is insufficient for identical dx. The source module explicitly records
this difference. Its stated performance/error measurements are not measurements
made by this study. The new rule preserves the KCP numeric stage only.

Forward native fwd675940a7a4fc0dba4b53 at `:500-520` forms the same ordered
four-tap preactivation, approximate SiLU and one BF16 output cast. The head
kernel computes only needed first-three-row dpre and writes each valid consumer
send slot. The main kernel also produces this rank output-owned FP32 dweight
partial; no hidden cross-rank allreduce occurs.

Backward has two further organizations requiring their own complete binding:
cooperative main plus grid-wide join and inline dweight merge, or ordinary
main with one reserved SM plus a following reduce launch. The native
cooperative representative at `:1560-1636` stores exclusive per-CTA partial
rows, executes this_grid().sync(), sums rows lane-strided, then XOR16/8/4/2/1
before final four-tap stores. This is not the ascending linear combine of
ordered_workspace_combine@1. The ordinary grid changes the partial partition,
so per-form determinism does not imply equal dweight bits between forms.
The [grid-join trace](trace-partial-grid-join-placement.md) now completes the
split/reduce representative and resource binding. Zero-history and inference
variants remain; the convolution family is not closed.

## Existing-rule scope and recovery

Pinned [causal_window_state@1](inputs/reuse-additions/causal_window_state@1/template.yaml)
uses width4 and history_capacity3 for a streaming register window, but its
ring/linear state commit has no rank row-routing or backward dpre protocol.
Pinned [history_suffix_pullback@1](inputs/reuse-additions/history_suffix_pullback@1/template.yaml)
routes gradients of an explicitly returned final history to current/initial
samples. This KCP operator does not return that differentiable history output;
its reverse payload is activation dpre for stencil edges. No substitution of
those suffix-cotangent fields for the distributed protocol is valid.

The [exact check](checks/check_document_stencil_halo_pullback.json) compares510
fragmented stencil cases to an independent full-document stencil and its
explicit transpose. It covers11294 transpose edges,1158 multi-consumer dpre
rows and900 multi-source histories, plus48 identifiable packed channel
coordinates. Rank labels include nonchronological layouts and empty ranks.
It checks exact local tap-partial sums against the whole-document tap gradient.
Native SiLU approximation, FP32 FMA/cast order, exchange progress and pool
lifetime remain implementation obligations; no GPU execution is claimed.
