# CUDA Hopper FA3 differences

This pass compares the retained hopper/ CUDA family with the
[CuTe Hopper representative](trace-cute-sm90-forward.md) and
[CUDA FA2](trace-cuda-fa2.md). hopper/setup.py and generate_kernels.py register
SM8x and SM90 variants, forward/backward/combine, optional split/paged/append,
FP16/BF16/FP8, varlen, head-size and cluster menus. Build-disable switches
remove capabilities; a helper's template availability is not an enabled path.
All paths below are relative to hopper/ in the [retained tree](inputs/sources.json).

## Shared designs with concrete bindings

| Source decision | Rule and field / explicit gap | Binding and consequential difference |
|---|---|---|
| flash_fwd_launch_template.h:35-100: architecture and scheduler selection | Existing output_tiling@2, compute_instruction_selection@1; persistent_tile_scheduler@1 where its actual policy matches | SM90 stages2 and TMA/WGMMA mainloop; SM8x uses register MMA and its separate tile/stage/Q-register menu. Dense noncausal SM90 uses static persistent, causal/local dynamic persistent, varlen a prepared dynamic scheduler, and fixed non-varlen split work uses single-tile scheduling. Do not import the later CuTe CLC work-claim backend by name. |
| mainloop_fwd_sm90_tma_gmma_ws.hpp:121,285-349,1141-1250 | warp_specialization@1, per_slot_handshake@1, multi_buffer_rotation@2, cross_iteration_software_pipelining@1 | Producer32 threads only for direct TMA Q/KV without V transpose; otherwise128. Two KV stages, QK(next) overlaps PV(previous), and V releases after its final asynchronous PV reader. Large-value cases may save a scale for correction before the next PV. This is the earlier pipeline family, not the distinct many-role SM100 graph. |
| flash_fwd_kernel_sm90.h:408-416; softmax.h:137-155 | online_rescaled_reduction@2 gain_log2=8 for FP8,0 otherwise | Q/K per-batch/KV-head descales multiply score scale before exp; FP8 exp weights use common gain256. Normalize O by the gained denominator, but divide the denominator by256 before taking log for LSE. V descale is a final output scale in mainloop:1241-1242,1339-1340. A score descale cannot be deferred through softmax like a value descale. |
| mainloop_fwd_sm90_tma_gmma_ws.hpp:1067-1078; utils.h:574 onward | Score-transform ordering / derivative binding gap | Softcap runs before support masking, since tanh(-inf) becomes finite. FP8 Q/K descales enter the tanh argument rather than being multiplied into its output afterward. This preserves the defined transformed score; generic exp2 approximation alone does not encode the whole nonlinear operator. |
| mainloop_fwd_sm90_tma_gmma_ws.hpp:1147-1190 | Existing absorbed-score decomposition from MLA trace, with native binding | Optional Qv*V^T accumulates into QK scores before one normalization. V's lifetime includes both score and output readers. Large V and the head64/value512 mixed RS/SS choices introduce heterogeneous consumer behavior outside a claim that all consumer WGs are interchangeable. |
| mainloop_fwd_sm90_tma_gmma_ws.hpp:535-539,693 onward | indexed_gather@1 plus source TMA/nonTMA binding | TMA paging requires page_size divisible key tile and no left padding. NonTMA page lookup handles more general addressing and append. Original sequence coordinates remain distinct from page IDs and effective lengths. |
| mainloop_fwd_sm90_tma_gmma_ws.hpp:1098-1119,1551-1698 | Rotary and mutating-cache ownership/publication gaps, as in FA2 | Producer stages new KV, consumer rotates/stores each valid new slice, then advances read state. Head packing remaps Q rotary positions as well as ordinary mask coordinates. Read-side gather and operand fusion do not establish concurrent append ownership. |

For a finite two-key row with raw scores(0,ln2), Q/K descale product2 changes
the ratio from1:2 to1:4. V descale3 instead multiplies the weighted output by3
without changing the probabilities. FP8 gain256 cancels from O but must be
removed from LSE. These substitutions distinguish the three scale roles;
native FP8 casts and exponent ranges still need their numerical bindings.

## FP8 V transpose and output permutation

mainloop_fwd_sm90_tma_gmma_ws.hpp:44-84 derives
Transpose_V=Is_FP8 and not V_colmajor; FP8 PV must be register/shared. Direct
FP16/BF16 V uses the ordinary orientation. The transpose path at :177-209,
715-790 uses ldmatrix-transpose, byte permutation and stmatrix into a second
shared view. Both key/value dimensions are divisible32 and either value width
or key tile provides the required64 grouping. Producer/transpose threads wait
for loaded V, write all transposed bytes, fence and join before publication;
the original load slot releases only after transpose reads finish.

The register byte step maps source bytes[0,1,2,3,4,5,6,7] to words
[0,2,4,6] and[1,3,5,7] via byte selectors0x6420/0x7531. The chosen
stmatrix layout deliberately permutes value columns to avoid bank conflicts.
Epilogue epilogue_fwd.hpp:237-243 restores output order with
utils.h:532-570, before FP32 partial storage for split mode or after final
conversion when unsplit. All-eight-label recovery is required, not a repeated
small-value test. In one shuffle quad with input pairs(U0,L0),...,(U3,L3),
that helper yields upper slots(U0,L0,U1,L1) and lower slots(U2,L2,U3,L3).
This is a bijection across lane/fragment slots. Full physical V-to-output column
correspondence remains a native realization obligation: the helper bijection
alone does not prove it is the inverse of the producer layout.

staged_relayout@1 describes register->shared->global epilogues and their
write/read lifetime. It does not express this shared->register->shared operand
transpose or the paired value-column permutation and cross-lane output undo.
tensor_axis_permutation@1 is only a rank-three stride view. Record the missing
paired operand/output permutation mapping explicitly rather than stretching
either existing rule into an unsupported realization. This source path does
not establish a separate incoherent/Hadamard preprocessing design.

## Scheduling, backward and combine differences

tile_scheduler.hpp:220-358 gives a dynamic atomic-counter scheduler:
initial work is blockIdx.x, later work atomicAdd(counter,1)+gridDim.x. The
initial range and claimed range do not overlap; producer/consumer handshakes
publish the same work identity. Static persistence strides by gridDim.x.
Varlen :593-686 decodes grouped counts with a warp prefix and31-batch search
windows; prepared metadata includes actual dynamic split count and virtual
batch mapping. flash_prepare_scheduler.cu:158-195 really sorts int4 batch
records in descending work order and remaps metadata accordingly. This differs
from the CuTe prepare path's unimplemented sort branch. A reused scheduler
needs that virtual-to-original map at every descriptor/output lookup; sorting
metadata alone cannot change the output batch identity.

flash_bwd_launch_template.h:30-80 chooses SM80 or SM90 backward and allocates
padded varlen FP32 scratch. It fixes cluster shape1x1, so the forward cluster
menu does not imply clustered backward. The SM90 mainloop reconstructs P,
forms dP/row-centered dS, retains dK/dV per KV output tile and combines dQ
through FP32 bulk adds or supported vector atomics (:954-992). Deterministic
dQ at mainloop_bwd_sm90_tma_gmma_ws.hpp:626-664 waits on row-specific rank
before the bulk add, releases after completion, and advances skipped local
rows so later owners do not wait forever. Causal order reverses key rank
relative to the row's admissible upper bound. This binds the same ordered
bulk-add/rank-map gap as CuTe SM90; FA2's owner-private dQ scratch is a
different deterministic alternative. Optional slicing is explicitly rejected
with deterministic mode at :268.

Forward split combine in flash_fwd_combine_kernel.h follows normalized-partial
O/LSE rescaled merging, matching the representation described in the
[CuTe combine trace](trace-specialized-backward-and-combine.md). Public combine
flash_api.cpp:1579-1637 checks FP32 partials, at most256 splits and FP32/BF16/FP16
output; standalone combine reuses the same state algebra with its own strides.
The FA3 public forward signature has no dropout parameter or replay descriptor;
inherited docstring mentions of dropout are not evidence for the FA2 mechanism
being connected here. Tail, mask, rotary, cache and FP8 restrictions are explicit
binding boundaries. The CUDA FA2/FA3 design families now have source-backed
dispositions, with native instruction/layout and the named extraction gaps
kept separate from benchmark or admission claims.
