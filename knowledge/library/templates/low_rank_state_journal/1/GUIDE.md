# low_rank_state_journal@1: checkpoint plus accepted decayed rank one journal

## When to consider

When recent recurrent state changes are low rank, retain accepted corrections instead of rewriting the full state every verify. Evaluate virtual state through journal contractions, append drafts outside all old-reader windows, and periodically fold only prior accepted history. Rebase gates and publish cursors once after all layers, then expose only the accepted draft prefix.

## Computation before and after

Project a checkpoint-plus-rank-one journal and periodically fold accepted history while speculative entries remain separate.

Let base checkpoint H0 precede accepted records j=0..P-1. For P>0 let Gp=G_(P-1), w_j=exp(Gp-G_j), bdec=exp(Gp); for P0 use Gp0,bdec1 and zero history. Represent H= bdec*H0+sum_(j<P) w_j*u_j*k_j^T. Project any required key/query x as bdec*(H0*x)+sum_j w_j*u_j*(k_j dot x), avoiding eager full-state materialization. Use that virtual H to compute T current draft outputs and feedback corrections under the declared chunk recipe. Append new normalized k,u,G at(base+P+s)%32 with one shared-key writer, disjoint from the old live window. If P<flush_min, keep checkpoint and use Gnew=Gp+local_G. If P>=flush_min, fold old P into H0 with one declared checkpoint rounding, use Gnew=local_G, and exclude all current drafts from this fold. After every layer/head finishes, acceptance a publishes one cursor update: no flush len=P+a; flush base=(base+P)%32,len=a. Rejected draft suffix remains stale outside the live prefix. An acceptance of zero after flush still materializes prior accepted history into checkpoint but adds no draft update. Read/write lifetime and checkpoint/gate frame publish together before the next verify.

Recover the original computation: Expand the journal projection and checkpoint fold; distributivity yields the same declared virtual state before and after rebase. Current correction prefixes are causal and independent of later draft suffixes, so publishing only a entries gives the accepted state. Folding is a representation change of old accepted history, not acceptance of current drafts.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| draft_tokens | compile / boundary | [4, 8] | Native HMMA draft window; runtime acceptance chooses a consecutive prefix after outputs are computed. |
| flush_min | runtime / software | {"lo": 1, "hi": 16} | Fold accepted history when P reaches this threshold; the lazy source default17-T folds only when another full draft window would exceed logical window16. |

## Coupled constraints and scope

FlashInfer GDN history journal on SM90+ HMMA, K=V128, T4/8, accepted window<=16, physical ring32. The earlier verify-only16-slot ring requires P+T<=16 and external flush. This version captures its journal algebra through the safer32-slot fused-fold binding; it does not claim that earlier wrapper directly accepts32-slot tensors. UMMA T1..8, FP16-u/BF16-k and stochastic checkpoint rounding are numeric/backend extensions. Raw-input replay, arbitrary nonlinear updates, ring eviction without fold and atomic multilayer commit are outside this version.

Scope predicate: `flush_min <= 17 - draft_tokens and (p.scalar_decayed_rank_one_state_and_logical_checkpoint_generation_defined) and (p.journal_keys_corrections_and_cumulative_gate_frames_match and p.inactive_ring_operands_finite_or_zeroed_before_mma) and (p.accepted_history_is_consecutive_and_speculative_suffix_not_published) and (p.ring_base_and_live_length_valid_and_append_window_disjoint_from_live_readers) and (p.unique_live_slots_and_one_shared_key_writer_per_key_group) and (p.flush_folds_only_previously_accepted_history_and_rebases_current_draft_gates) and (p.all_layers_and_head_readers_complete_before_one_cursor_commit) and (p.journal_correction_gate_and_checkpoint_fold_precision_admitted) and (p.threshold_shape_target_dtypes_and_rounding_mode_implemented)`.

- Precondition `pre_state` (implementation): `p.scalar_decayed_rank_one_state_and_logical_checkpoint_generation_defined` — A V-major represented state is a base checkpoint plus ordered scalar-decayed rank-one corrections. Each u_j was computed against the same preceding virtual state from its matching generation; u is not raw V and cannot be recomputed from another checkpoint without replaying feedback.
- Precondition `pre_frames` (implementation): `p.journal_keys_corrections_and_cumulative_gate_frames_match and p.inactive_ring_operands_finite_or_zeroed_before_mma` — Store normalized keys k, correction vectors u and cumulative log gates G relative to the current checkpoint. G_j includes decay through insertion j. Empty history has G_anchor0 and no live contributions; invalid slots are masked before exponentiation, not by multiplying stale data by zero afterward. Inactive K/U lanes must remain finite (for example from zero-initialized finite ring records) or be explicitly zeroed before MMA; a zero weight does not sanitize NaN/Inf payloads.
- Precondition `pre_retention` (implementation): `p.accepted_history_is_consecutive_and_speculative_suffix_not_published` — P entries are previously accepted history. T draft corrections append speculatively; accepting a consecutive a in[0,T] publishes only that prefix, leaving stale suffix entries outside the live window. Diagnostic acceptance counts cannot replace a.
- Precondition `pre_ring` (implementation): `p.ring_base_and_live_length_valid_and_append_window_disjoint_from_live_readers` — A shared32-slot physical ring has live length0..16, base0..31, and logical row j maps to(base+j)%32. Appends target(base+P+s)%32 for0<=s<T, always after the old window and disjoint from all old-window readers across grouped value heads.
- Precondition `pre_ownership` (implementation): `p.unique_live_slots_and_one_shared_key_writer_per_key_group` — Per-request mutable slots are unique, value-head u/G/checkpoint slices are disjoint, shared normalized-key rows have one elected writer, and all readers agree on the same key-group/head mapping and generation.
- Precondition `pre_flush` (implementation): `p.flush_folds_only_previously_accepted_history_and_rebases_current_draft_gates` — When P>=flush_min, fold exactly old P entries into the checkpoint, not current drafts. New draft entries still append after old P, but their G values restart from zero relative to the folded checkpoint. Without flush, new G includes old anchor G_P.
- Precondition `pre_cursor` (implementation): `p.all_layers_and_head_readers_complete_before_one_cursor_commit` — Kernel bodies observe unchanged old base/P. After every layer/head completes, commit cursors once: flushed base becomes(base+P)%32 and len=a; nonflushed base stays and len=P+a. No layer commits shared cursors early; future verify waits for complete checkpoint/journal publication.
- Precondition `pre_numeric` (implementation): `p.journal_correction_gate_and_checkpoint_fold_precision_admitted` — Normalization stage, stored k/u/G precision, exp differences, scalar weighting, MMA products, checkpoint rounding and optional stochastic state rounding have an explicit admitted recipe. Mathematical representation equivalence is not native bitwise equality across folds/backends.
- Precondition `pre_backend` (implementation): `p.threshold_shape_target_dtypes_and_rounding_mode_implemented` — Selected target supports T and chosen ring/state dtypes and rounding mode. This first binding is T4/8, logical window16 and physical32, with flush_min<=17-T. UMMA broader T/mixed-u/stochastic-rounding menus require their own numeric/RNG binding.
- Constraint `c_threshold` (implementation): `flush_min <= 17 - draft_tokens` — No request may be unable both to append within logical capacity and to flush. Physical32 additionally retains old16 plus speculative T without overlap.

## Local example and misuse

For a local instance of `store`, identify the object `checkpoint_plus_accepted_decayed_rank_one_journal` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/low_rank_state_journal@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/low_rank_state_journal@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Low-rank normalized-key, correction and cumulative-gate ring beside one checkpoint; old readers and new append use disjoint windows.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `virtual_state_identity` (proof): Checkpoint plus decayed low-rank journal equals its declared sequential represented state and yields the matching key/query projections.
- `accepted_prefix_and_rebase` (test): Flush rebases old accepted history only; all accepted-prefix outcomes, including zero, retain correct new gate frames and cursors.
- `shared_reader_ring_safety` (test): Modulo32 append locations cannot overwrite the old live window needed by sibling value-head CTAs; shared keys have one writer.
- `numeric_and_cursor_publication` (test): All native correction/fold/RNG stages meet the numerical contract and shared cursors advance once after all layer/head users finish.

Correction caching freezes prediction rounding; folding introduces checkpoint rounding and may differ from raw-input sequential replay. Stale gate exponentiation, early shared cursor updates or folding unaccepted drafts changes future state.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
