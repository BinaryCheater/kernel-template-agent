# Retain entry states to replay local training tapes

When recurrent training tapes dominate saved memory, retain the nonlocal
fragment-entry states and regenerate local tapes in backward. The entry states
form the dependency boundary: replaying local work does not require repeating
the forward boundary exchange, but backward still needs reverse boundary adjoints.

[boundary_seeded_tape_replay@1](delta/boundary_seeded_tape_replay@1/template.yaml)
and its [outline](delta/boundary_seeded_tape_replay@1/proof.yaml) expose three
retention choices. This is distinct from splitting a forward recurrence into
independent affine summaries: the choice controls which artifacts survive into
backward and which dependency subgraph is replayed.

The [retained source](inputs/sources.json) is the deterministic KCP GDN package
under `flashinfer/experimental/cake_gdn_det_train_kcp/`; `gdn_train.py` is a
compatibility facade. The selected scope is SM100/103/107, 64-token local chunks,
BF16 Q/K/V, FP32 activated gates and native grouped Q/K heads with HV divisible
by Hqk. Optional raw-Q/K normalization adds FP32 rstd checkpoints and its own
backward stage. No native training or distributed execution was run here.

| Concrete decision | Rule field / binding | Source and evidence |
|---|---|---|
| Keep all local forward tapes and configured boundary artifacts | forward_retention=full_tape | `cake_kcp.py:584-599,5048-5080`: full save binds FWD_TAPES; saved boundaries include S_in, optional owner receive and M_seg, plus norm statistics |
| Keep S_in and regenerate local factors, carries and tapes | forward_retention=entry_boundaries | `:4185-4256`: fwd_prep, local b_seg and optionally M_seg, carry merges, then seeded tape walks; no forward exchange or crossing-owner merge |
| Keep no forward-derived artifacts and rerun boundary resolution | forward_retention=nothing | `:4276-4355`: saved=None calls _forward(save=False,replay=True) before backward; light mode actually keeps no local tapes, rather than a residual cheap-tape subset |
| Preserve input, precision, normalization and partition generation | pre_inputs / pre_tapes | `:5081-5116` checks context and modes, not tensor contents; caller must maintain unchanged/versioned inputs. Matching scalar recurrence is insufficient for BF16/TF32 tape identity |
| Omit forward output only when this replay has no output consumer | pre_output | `:4022-4066` direct backward replay omits fwd_o. Torch checkpoint recompute follows a separate frame/liveness check and cannot inherit this omission unconditionally |
| Keep reverse boundary communication in every retention mode | pre_reverse | `:4276-4355` selects saved/replayed operands, then starts the adjoint offset and reverse-exchange path regardless of forward retention |
| Keep tapes through their last reader, with fresh owned retained storage | pre_lifetime | `:4945-4998` keeps at most one recompute-produced tape set and binds fresh tensors; _kept_release after the gradient program releases the last readers |
| Caller recompute handle retains S_in across checkpoint replay | entry_boundaries ownership | RecomputeHandleDetKcp is an external retained-boundary owner, not zero-artifact replay. Its layer/microstep identity, graph restrictions and separate backward exchange remain required |

## Concrete native witnesses

The literal `cake_jit.py` module registry chooses
`cake_gdn_det_train_01f0e0eab26c58ff3953_kcp` for the full-V forward tape
walk. Its `csrc/*_kernel.cu:215,323-340` accepts indexed h0, resolves the
segment/head/chunk range and reads the FP32 seed when the index is valid.
`:378-397` walks chunks forward; `:807-849` publishes h_tape and v_new with
TMA stores. Invalid seed indices denote declared zero starts, not arbitrary
rank-local replacements for a crossing document state. The V64 sibling changes
the value tile, not the retention boundary.

The reverse tape representative
`cake_gdn_det_train_8b193b1101cc03b7d704_kcp` accepts indexed dht and resolves
the segment/head at `csrc/*_kernel.cu:210,323-337`. It walks chunks in reverse
(`c=nc-1-rc`, `:378-384`), carries FP32 adjoints, emits BF16 dh_tape and
dv_new, and reads q/k/w/do_g in reverse chunk order (`:384-635`). The issuer
`:638-821` performs the K-by-dh value-adjoint product and Q/do_g injection
and W/weighted-value-adjoint correction with ready/empty events. Thus forward
seed retention supplies forward operands; it cannot supply a future boundary
adjoint that has not yet been computed. The exact transport composition and
communication tables need their own trace.

## Recovery and limits

The [exact check](checks/check_boundary_seeded_tape_replay.json) compares three
retention modes over 51 cases, including empty local tapes, interior empty
fragments and nonzero crossing seeds. It uses a scalar delta recurrence with
feedback and an independent affine-summary route for reconstructed entries.
Manual reverse gradients match 440 derivatives of an independently expanded
polynomial loss. Symmetric differences are exact rational polynomial
identities here, rather than an approximate small-step numerical check.

Counterexamples distinguish a zero crossing seed, changed replay inputs and
omitted terminal boundary adjoint. These checks establish the dependency and
scalar-gradient premises. Native factor/tape rounding equivalence, rank
exchange order/progress, checkpoint frame liveness and last-reader storage
release remain the corresponding implementation obligations. The three modes
exclude input/autograd-context memory and transient/static workspace from the
retained-artifact byte effects.

For a concrete composition, state_independent_chunk_factors@1 supplies the
recurrent local preparation concept, affine_piece_prefix_state@2 supplies
ordered forward boundary propagation, and this rule chooses whether those
boundaries or their full local consequences survive into backward. The first
two do not supply the retention variable or backward-adjoint contract.
