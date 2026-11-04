# Transport future recurrent adjoints through reverse affine chains

When a recurrent training document spans rank fragments, local backward cannot
finish from a rank-local zero adjoint. Summarize each fragment backward with a
zero exit adjoint, then transport its future contribution through transposed
forward maps in document-predecessor order. The source offers head-owner
exchange and a restricted short neighbor-hop schedule.

[reverse_affine_boundary_transport@1](delta/reverse_affine_boundary_transport@1/template.yaml)
and its [outline](delta/reverse_affine_boundary_transport@1/proof.yaml) add
that distributed choice. The [retained source](inputs/sources.json) is the
SM100/103/107 deterministic KCP GDN package under
`flashinfer/experimental/cake_gdn_det_train_kcp/`. This trace follows the active
`cake_kcp.py` table builder, not just the older helper `cake_tables.py`:
the two helpers use different stored-map orientations in their adj_b1 binding.

For column state, let fragment f have S_out=M_f*S_in+B_f. Its local backward
with zero exit adjoint yields d_f. Given future exit adjoint lambda_out, its
entry adjoint is M_f^T*lambda_out+d_f. Document terminal adjoint is zero in
this source. Thus the last fragment contributes d_last to its predecessor;
each earlier fragment receives only the contribution strictly after itself.
This is an exclusive future boundary, not an inclusive local derivative.

| Concrete source decision | Rule field / binding | Evidence |
|---|---|---|
| Gather d_f to head owners, merge backward and return dS_out | boundary_transport=head_owner_exchange | `cake_kcp.py:4276-4416`: b1 all-to-all, wait, adj_b1/adj_c1 owner merge, b2 all-to-all, wait before crossing consumers |
| Saved forward map versus resend map beside d_f | transfer_source=saved_forward_map / resent_local_map | m_source selects d versus dm b1 form. Saved c1 contains M_f^T; resent b1 contains M_f. Active FAMILIES at `:1898-1993` uses tm=1 for adj_c1, tm=0 for adj_b1 |
| Physical tile orientation differs from mathematical operator | pre_maps | `:1691-1693`: transpose_m=0 consumes storage Mop^T; transpose_m=1 consumes Mop. Both adjoint bindings therefore apply mathematical Mop=M_f^T. The flag alone does not mean transpose the forward map |
| Reverse fragment order and predecessor destinations | pre_order / boundary_placement | `cake_kcp_plan.py:1774-1815` visits fragments from last to first and addresses b2_send at the predecessor's crossing-out slot. Last-fragment unused map has sentinel -1 |
| Initialize at d_last, then fold only interior fragments | transpose_chain | Active `cake_kcp.py:2213-2243` sets x0=d_off[e0], constructs elements e0+1..e1-1 and writes exclusive boundaries via prior output offsets. Two fragments need no matrix application at the owner |
| Propagate completed adjoints along document neighbors | boundary_transport=neighbor_hops, transfer_source=resent_local_map | `:3709-3799`: first hop sends d_last, later hops apply M_f^T*dS_out+d_f locally; send to prev_rank, receive from next_rank, wait then scatter into b2_recv |
| Bounded neighbor menu | pre_ring | Source options require n_d_max-1<=2 and save_transfer=False. Ring here follows document predecessor links, not an arbitrary cyclic rank reduction. No broader hop support is inferred |
| Per-head owner capacity and split holes | pre_order / pre_owner | `cake_kcp_plan.py:1245-1259,1453-1464,1595-1622`: Hg=ceil(HV/W), owner(h)=h//Hg; last owner may have unread padded tile slots, later ranks may own no heads yet participate in collectives |
| Local subsegment adjoints use the same reverse chain | pre_scope | P>1 builds zero-exit offsets for all segments, tmap_d summaries and dcarry_end/dcarry_cross with transposed map families; source end fragments start at zero |
| Run document-end tape walks during b2/hop1, exactly once | pre_lifetime | `:3709-3799,4390-4445`: ends need no future message. ended_early/merged_end suppress duplicate work; dv_new aliases dv_intra, so replaying an already executed walk changes its operand |
| No crossing fragments means no boundary exchange | pre_scope | tb.N_out==0 bypasses transport but still runs local adjoint carries, tape walks and gradients |

## Concrete merge body

`cake_jit.py` registers c8/c16 affine merge variants. The c8 witness
`csrc/cake_gdn_det_train_c6958a0a4ed5cb55a9a1_kcp_kernel.cu:66-100`
assigns a CTA to one chain and eight columns. `:105-164` initializes shared
state from x0 or zero, publishes the initial entering state when requested,
and handles a chain with no interior element by writing x0 to y_final.
`:165-435` folds one map/offset at a time through explicit FP32 __fmaf_rn
operations, then publishes entering/last state before the next element.
`:435-1028` provides the other stored-orientation path. These are concrete
state publication and arithmetic witnesses, not a network or timing result.

## Reuse boundary and check

[affine_piece_prefix_state@2](delta/affine_piece_prefix_state@2/template.yaml)
binds local reverse affine algebra after transposing the adjoint into row
form: R_next=R*M_f+d_f^T, with reverse document order and shared_fixup.
Its pre_affine and pre_prefix_supply cover ordered composition and exclusive
entry maps, but its scope explicitly excludes distributed collectives. It
has no owner-versus-neighbor transport or saved-versus-resent transfer field.
Those choices and their waits/orientation contract belong to the new rule.
[boundary_seeded_tape_replay@1](delta/boundary_seeded_tape_replay@1/template.yaml)
chooses forward artifact retention; it does not remove this reverse exchange.

The [exact matrix check](checks/check_reverse_affine_boundary_transport.json)
compares48 owner-chain cases,24 supported short-hop cases and96 independently
computed seed derivatives of a forward loss. It checks1224 live head-owner
slots including partial/empty owner groups. Noncommuting matrices distinguish
wrong map order, missing transpose and inclusive/self boundary use. Saved
transposed tiles and resent ordinary tiles resolve the same mathematical
operator in the check. It does not implement fake collectives or establish
GPU FP32/TF32 equality, progress or alias/storage lifetime.

Remaining training designs are raw-normalized/grouped gradient organization,
checkpoint-frame output/tape handoff, raw gate epilogues, and the distinct
causal-convolution training/halo paths. This trace does not close training or
the full library.
