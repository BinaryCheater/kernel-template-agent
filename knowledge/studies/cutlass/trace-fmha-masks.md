# FMHA masks, empty rows and crossed read-retirement channels

When score and probability tiles alias tensor memory, recover the **last score
read before probability overwrite**, including pipeline prologue/tail events.
Retrieve `physical_storage@1(level=tmem,buffer=shared)` for its per-region lifetime
inequality. Do not infer a full/empty credit protocol from a barrier object's name.
For structural masks, prove both skipped tiles are identity-only and every
unmasked loop tile is valid for visible rows before reusing mask-pruning rules.

This slice continues [FMHA pruning](trace-fmha-pruning.md). Source ranges below
refer to forward `fmha.py` and `examples/python/CuTeDSL/helpers/fmha_helpers.py` in
the [retained tree](inputs/sources.json). FMHA forward is still active; the full
TMEM/MMA lifetime graph, cross-work retirement, resource/scheduler paths and
backward remain outstanding.

## Mask classification and actual host selection

`FusedMask.get_masked_info`:927–1008 returns five counts: start, end, leading
masked, unmasked and trailing masked. The forward softmax executes these three
segments in ascending order. Helpers derive the window offset as0 for ordinary
alignment or seqlenK−seqlenQ for bottom-right alignment; element masking applies
inclusive window bounds and residual Q/K limits. Forward's helpers use the full
CTA M extent, which contains two softmax groups; tests therefore require an
unmasked tile to be valid for all **live** query rows of that CTA.

[Check](checks/fmha_masks.py) executes retained enum/helper methods at2742 mask
classification domains/440716 distinct score coordinates. This includes small
exhaustive geometries and the actual CTA256×128 shape at Q257/K129 for residual,
causal and bottom-right window variants. In selected host domains every kept coordinate retains its unique
score and masked coordinates become−inf; no nonidentity
tile is omitted and no unmasked iteration contains a masked element for live
queries. Counts need not prove a padded query row is valid: it has no output
permission.

Two implementation boundaries matter:

- Window helper with both window sides None performs no residual mask itself.
  Directly using it on a partial K tile leaves padded scores unchanged. Host
  `run`:3685–3697 chooses residual mask instead for tuple KV lengths or a fixed
  KV length not divisible by tileN. The recorded direct-helper counterexamples
  are outside that selected host branch, not evidence that the host forwards
  unmasked K tails.
- Existing `masked_tile_pruning@1.pre_neutral` supplies exact skip legality and
  its three obligations match these classification checks. Its first-masked/
  remaining-unmasked program and one peeled-step count do not fully express a
  leading-mask/unmasked/trailing-mask schedule. A solver-facing variant/binding
  remains to be defined rather than claiming the old fields recover all three
  segment boundaries.

## Empty visible rows are not ruled out

Host's nested `check_seqlen_valid`:3704–3718 rejects an interior row only when its
computed endpoint difference is exactly0, with the first/last rows exempt. It
uses inclusive element windows but an endpoint subtraction without the matching
inclusive count. The retained function:

- accepts Q3/K1 with bottom-right causal masking (left None,right0), although
  visible query rows0 and1 have no allowed key;
- rejects Q3/K3 with the nonempty diagonal window(left0,right0), since its
  interior row has endpoint difference0.

These are predicate/element-map results, not native output measurements. The
helper sweep includes960 empty visible rows before the host validity filter;
that count is not a count of accepted public configurations. The explicit Q3/K1
case demonstrates that host filtering is not a proof that every visible row has
positive mass.

All-masked scores leave row sum0 in real arithmetic after the safe-reference
substitution. Finalizer `correction_epilog`:3220–3231 directly divides by row_sum
and computes log(row_sum) for LSE; this slice establishes no defined zero-mass
output policy. A finite sink can supply denominator mass but still needs the
common-gain correction from the [frame trace](trace-fmha-frames.md).
`online_rescaled_reduction@2.final_denominator_positive_finite` and the pruning
rule's positive retained mass therefore remain essential rejecting conditions.

## Aliases and offset channel accounting

Constructor offsets and `make_tmem_tensors`:993–1029 place S0 at column0, S1 at128,
P0 at160 and P1 at32. Skip flags for P0/P1 are at136/8 respectively. For the
selected128-row, FP16-P witness, a nominal64-column P0 lies inside S1's128-column
region; P1 lies inside S0. Native fragment/layout composition must separately
establish these footprints. The offset test does not admit other P/output widths.

| Source event | Permission supplied | Boundary |
|---|---|---|
| `softmax_step`:2306–2309,2384–2385,2451–2452 | S0/S1 capture completes before committing its in-place producer channel. | `physical_storage@1.pre_lifetime` supplies last_read+delta≤next_write. Native capture/fence semantics remain required. |
| Pipeline construction:926–947 | Two `PipelineAsync` channels, one stage each,128 producer and128 consumer lanes. S0 publishes to P1; S1 publishes to P0. | These channels use full-barrier arrivals/waits as read-retirement notifications; no producer empty-acquire appears on these paths. |
| Stage1 prologue:2878–2880 | Consume S0's initial read-retirement event before entering its first iteration. | Aligns the P1 overwrite with the **next** S0 read rather than the same iteration's event. |
| Both ordinary conversion:2129–2156 and skipped path:2566–2580 | Wait peer channel before writing P or its skip flag; skipped arithmetic still waits/publishes the pipeline generation. | Missing credit-release is not by itself a bug: no corresponding empty credit is acquired. Reciprocal sequencing must prove safe reuse instead. |
| Stage0 tail:2914–2916 | Send the additional S0-channel event after its last iteration. | Matches stage1's final wait; removing it strands the last P1 step in the abstract model. |

`PipelineConsumer.wait_and_advance` in `pipeline/sm90.py:1403–1425` returns an
immutable handle and advances; it does not implicitly release empty credits.
The source intentionally discards that handle on these notification channels.
The old `per_slot_handshake@1.program` explicitly waits on empty and releases it,
so setting its `full_signal` field cannot express this different incidence or
prologue offset. A read-retirement channel mapping remains a distinct interface
gap; the per-region storage invariant is reused only conditionally.

A bounded abstract model explores351 states over stream lengths0..12. Each valid
stream has one completed terminal state, no stranded waits and at most one
unconsumed generation per channel. Removing stage0 tail strands a wait; removing
stage1 prologue yields14 unsafe-generation/pending events for length3. For K
iterations, S0 channel has K+1 publications/waits (including boundary events),
while S1 has K. The model is per work item and assumes score-read events are made
available by the unmodelled MMA pipelines. It does not prove native scheduling or
cross-work S/P reuse; those edges must be checked in the full graph.

[Result](checks/fmha-masks-result.json) retains source offsets, mask cases,
selected/direct-helper distinctions, host empty-row predicates and the abstract
channel evidence. No new candidate is added in this slice. Next: encode the
three-segment mask choice, reconcile cross-work MMA/TMEM ownership and complete
forward resource/layout scheduling before moving to backward.
