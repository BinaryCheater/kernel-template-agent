# Separate dead checkpoint output from recompute-tape handoff

When non-reentrant checkpoint replay regenerates forward tapes that backward
will immediately need, two choices arise: skip an output stage with no reader,
and retain those tapes rather than generating them again. They have different
correctness and memory conditions, even though source auto modes use the same
last-saver frame proof to select them.

[checkpoint_recompute_liveness@1](delta/checkpoint_recompute_liveness@1/template.yaml)
and its [outline](delta/checkpoint_recompute_liveness@1/proof.yaml) capture
both choices. The [retained source](inputs/sources.json) is
`flashinfer/experimental/cake_gdn_det_train_kcp/cake_kcp.py`. This is host
lifetime/launch orchestration around the concrete forward and backward programs
in [tape replay](trace-boundary-seeded-tape-replay.md); no new GPU arithmetic
implementation is inferred.

## Output deadness

Source `:778-883` documents and implements the frame adapter. The backward
reads dout, saved operands, saved boundaries and tapes; it does not read the
forward output itself. A non-reentrant checkpoint replay exists to repopulate
saved-tensor holders. If this executable node saves exactly the remaining
holders and early stop is enabled, replay stops immediately after these packs,
so no subsequent region operator reads its output. Its output tensor can stay
allocated with the same shape while fwd_o is omitted and contents remain
unwritten. The proof requires all of the following:

- Grad mode and a requiring-grad operand made the node executable, as checked
  before entering autograd Function.forward. Non-executable nodes save no
  tensors; a coincidentally matching holder count can belong to a later node.
- The top saved-tensor hook is the recognized internal recomputation hook,
  with target_frame_ref and gid. A hook in original forward, a user hook,
  reentrant checkpoint or merely being inside backward does not establish this.
- The recognized frame has early_stop=True and remaining holders equal the
  exact number this node will save next. More holders indicate later savers;
  fewer indicate mismatch, not permission to skip work.
- The versioned frame adapter recognizes the runtime release/protocol. The
  source accepts release2.14.1 with local +build suffix stripped, but does not
  accept arbitrary prereleases. This is its source compatibility menu; this
  study has not executed the private torch protocol.

`_saved_count` at `:4924-4942` includes five operands, S_in, configured owner
transfer/local segment map, any full-mode tapes for live tokens and optional
qk_rstd even for zero local tokens. `_fwd_body:4945-5045` caches the proof by
saved count, allocates output as usual and conditionally launches fwd_o. Direct
forward and captured graphs do not receive this elimination. An unavailable
proof chooses the ordinary output program.

## Handoff and peak memory

| Concrete choice | Rule field / binding | Evidence |
|---|---|---|
| Discard recompute tapes and replay local preparation/walk in backward | tape_handoff=replay_in_backward | Source False; ordinary light saved-boundary replay remains the fallback |
| Keep tapes from a valid recompute for matching backward | tape_handoff=retain_for_backward | Source True; runtime task identity is required, but the last-saver frame proof is not the selector. Keeping through a later projection/MLP backward may raise peak memory |
| Keep only if the executable last-saver frame proof succeeds | tape_handoff=last_saver_only | Source auto at `:4973-4988`; unreadable frames and later savers choose local replay |
| Compute output or skip only with the no-reader proof | recompute_output=compute / elide_with_frame_proof | Source always/auto; this choice is independent of tape retention |
| Retained tapes leave the writable workspace arena | pre_storage | `:4990-5045`: owned tape tensors, normalized Q/K rows and optional stage_m plane remain referenced while arena bindings are removed; a later workspace forward cannot overwrite them |
| Handoff does not add save_for_backward tensors | pre_recompute | `:5360-5475`: saved sequence stays five operands plus saved-set tensors; the kept set travels separately through the task registry so original/recompute holder counts agree |
| Identity includes task and operand/boundary pointers plus dimensions | pre_identity | `_kept_key:931-958`; owning tensors keep those pointer generations alive. Equal addresses without lifetime/task identity are not a reusable cache key |
| One unconsumed entry, eviction before new materialization | pre_bound | `:961-1002,4988`: old entries are cleared before new handoff work and on put; take removes the entry once. Missing/evicted nodes use ordinary local replay |
| Explicit last-reader release | pre_storage | `_kept_release:4256-4274` clears after gradient program8, optionally readopts compatible post-forward arena tapes only at matching geometry; saved owner pools and local maps retire at their earlier last readers |
| Unconsumed and failed-task cleanup | pre_bound | Task-end callback drops an unused entry; source notes failure may suppress callbacks, so later eviction also bounds leftovers. Callback success alone is not a universal failure-cleanup proof |
| Backend restrictions | pre_backend | Kept validation at `:5150-5205`: light save, eager graph off, nonzero local tokens, exact tape names/shapes/geometry and optional norm/stage_m presence. Captured graph storage has a different lifetime |

Source auto retention prevents tapes spanning later-saver backward work. It
does not establish a whole-model peak bound: branches, allocator behavior,
other gradients and operations with no saved tensors still require their own
memory accounting. The source numerical recipes stay unchanged by handoff;
using a tape from another task/generation would break that premise.

## Recovery and reuse boundary

The [logical schedule check](checks/check_checkpoint_recompute_liveness.json)
explores384 replay schedules;16 satisfy executable-node/exact-holder/early-stop
deadness. Counterexamples show equal counts without an executable node,
early-stop disabled, a later saver and an altered saved-tensor sequence.
An illustrative lifetime schedule has peak120 memory units when40 tape units
span80 later-backward units, versus80 with later local replay; a last-saver
schedule has60 units. These are chosen event sizes demonstrating the lifetime
mechanism, not measured bytes or latency. It also checks one-entry eviction,
one-use retrieval and graph-task key separation. It is not a fake torch
execution engine or a validation of hook ABI/callback/native tape behavior.

[boundary_seeded_tape_replay@1](delta/boundary_seeded_tape_replay@1/template.yaml)
provides the same local replay alternative and retained entry states. Its
pre_output requires no output consumer and pre_lifetime requires last-reader
ownership, but it has no executable-frame proof, output-elision choice or
bounded handoff policy. The new rule supplies those missing controls.
[shared_normalization_gradient_merge@1](delta/shared_normalization_gradient_merge@1/template.yaml)
explains why normalized rows and the original saved rstd must remain correctly
paired when norm tapes join the handoff.

Raw gate epilogues and causal-convolution train/backward/halo remain separate
unfinished training designs. Native torch frame integration, nested-checkpoint
behavior and task-failure cleanup remain named obligations of this proposal.
