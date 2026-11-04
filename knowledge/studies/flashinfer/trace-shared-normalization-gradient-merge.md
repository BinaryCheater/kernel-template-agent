# Merge shared-row gradients before normalization pullback

When one normalized Q/K row serves several value heads, accumulate all of their
FP32 incoming gradients before applying the shared raw-row normalization
pullback. Choose a group-owned accumulator or materialized FP32 partials;
keep the group order, raw/statistic generation and final cast boundary fixed.

[shared_normalization_gradient_merge@1](delta/shared_normalization_gradient_merge@1/template.yaml)
and its [outline](delta/shared_normalization_gradient_merge@1/proof.yaml)
record this choice. The concrete [retained source](inputs/sources.json) is
`flashinfer/experimental/cake_gdn_det_train_kcp/` on SM100/103/107. Raw Q/K,
normalized forward operands and final raw gradients are BF16; saved inverse
norms and incoming head partials are FP32. Hqk divides HV and each Q/K head
serves r=HV/Hqk consecutive value heads. The channel extent is128.

For one raw row x, forward computes rstd=1/sqrt(sum(x*x)+1e-6) and supplies
BF16(x*rstd). The source backward convention is the smooth normalization
pullback, using y=FP32(x*rstd):

    D = ordered_fp32_sum(D_head)
    s = complete_row_dot(D, y)
    dx = BF16(rstd * (D - y*s))

This convention treats the activation cast as the source's narrow checkpoint;
it is not a mathematical derivative of the discrete BF16 quantizer. The
unrounded y needed by the pullback differs from the stored forward BF16 row.

| Concrete source decision | Rule field / binding | Evidence |
|---|---|---|
| Persistent owner is (chunk,QK-head), not an independent value-head writer of dq/dk | gradient_merge_storage=owned_accumulator | Native bwd_dqkg_l2n `csrc/cake_gdn_det_train_3e6562c2f739184a3765_kernel.cu:600-655` resolves group ticket, loads saved rstd and loops hl=0..r-1 with hv=hq*r+hl |
| First head initializes TMEM sum, later heads add in fixed FP32 order | pre_order / pre_owned | Native `:2287` and `:2719` select group_reduce and hl==0; no atomic fan-in. Ticket completion can vary across groups without changing addition order within a shared row |
| Write FP32 per-value-head slots then merge through host tensor orchestration | gradient_merge_storage=fp32_partials | `cake_kcp.py:3223-3283` sets group_reduce=0 and dq_part/dk_part; `:3299-3333` reshapes [T,Hqk,r,128], initializes consumer0 then adds j=1..r-1 |
| Apply one norm pullback to the full sum before the sole BF16 cast | pre_stage | Same host path calls cake_l2n.l2n_pullback then converts/copies to output. Native fused sibling applies it after its head loop and before BF16 stores |
| Raw Q/K plus saved rstd, not forward rounded qn/kn | pre_generation | Native `:2911-2998` stages raw rows and saved factors, forms products using mul_f32_rn_noftz. Host passes original ops q/k and rstd_in, not the normalized reader view |
| Complete128-channel dot uses an explicit tree | pre_dot | `cake_l2n.py:l2n_dot` halves each32-channel quadrant then adds ((S0+S1)+S2)+S3; native `:2990-3670` performs matching lane/warp reductions and publishes quadrant totals |
| Separate product, subtraction and final scaling without FTZ | pre_dot | Native `:3670-4200` loads row dot and saved r, forms y*s, sub.rn(D,product), mul_f32_rn_noftz(r,residual); final gradient stores through `:4664` use __float2bfloat16_rn |
| Forward statistics and checkpoint map have their own tree | pre_stage | fwd_prep_l2n representative `cake_gdn_det_train_09ade88178a6d04c8619_kernel.cu:390-573` does eight square FMA steps per lane, XOR8/4/2/1 reduction, +1e-6, sqrt.rn and __fdiv_rn; normalized row stores at `:575-637` |
| Full-save mode rebuilds normalized operands from saved r | pre_generation | `cake_kcp.py:4335-4339` launches qk_apply; light replay or kept tapes supplies matching normalized rows. `cake_l2n.py:qk_apply_ref` specifies BF16(FP32(raw*r)). Recomputed statistics are not a substitute without a stage-equality premise |
| Blocked host orchestration changes temporary size, not per-token order | pre_workspace | HOST_L2N_BLOCK_TOKENS=2048 and host_group_reduce process separate token blocks. This path still executes torch tensor arithmetic on the GPU; host does not mean CPU numerical execution |
| Packed QKV gradients keep separate token strides | pre_output | dqk_tstride and dv_tstride are explicit launch arguments; raw q/k strides and normalized operand strides have separate identities, and chunk_len masks valid tokens |

The literal registry selects native bwd_dqkg_l2n with384 threads and227328
shared-memory bytes, fwd_prep_l2n with192 threads and113664 bytes, and qk_apply
with256 threads. The ordinary no-norm bwd_dqkg sibling shares grouped fan-in
but omits the raw-row Jacobian. These are compiled resource records, not
measured occupancy or speed claims. Its fused/no-norm case is the same
owned ordered sum with the pullback disabled; the new rule's normalized scope
keeps that semantic distinction explicit.

## Existing-rule boundary

[ordered_workspace_combine@1](delta/ordered_workspace_combine@1/template.yaml)
provides the fixed ascending FP32 merge principle, but its split_count and
floor ranges partition reduction steps; its current scope is grouped weight
gradient tails. A value-head group is a consumer fan-in, not that floor-range
partition. It also lacks a raw normalization Jacobian between the sum and
cast. No direct binding to its floor-range fields is claimed here.

[boundary_seeded_tape_replay@1](delta/boundary_seeded_tape_replay@1/template.yaml)
provides saved/replayed operand generation and lifetime. It does not identify
which rounded normalization stage supplies the pullback or where head
partials merge; the new rule owns those decisions. Forward fusion may
supply normalized operands once per shared row, but cannot replace the saved
raw/statistic identity with the narrow activation.

## Focused recovery

The [check](checks/check_shared_normalization_gradient_merge.json) compares48
exact rational projection-form pullbacks against an independent dense
Jacobian r*I-r^3*x*x^T, and32 FP32 owned/materialized sum-and-pullback cases.
These checks treat the declared saved r as an operand; they do not validate
its native computation. Finite-format counterexamples show differences from
using BF16 normalized rows, casting per-consumer partials early, reordering
consumer additions, or changing the channel-dot quadrant tree. The Python
FP32/BF16 stage evaluator is logical evidence, not GPU execution.

Native TMEM pipeline equivalence, no-FTZ behavior, forward statistic identity
and saved-r last-reader lifetime remain implementation obligations. Raw gate
epilogues, checkpoint-frame handoff and causal-convolution training/halo are
still distinct unfinished training designs.
