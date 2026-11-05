# online_rescaled_reduction@2: running reduction state

## When to consider

Stream a softmax-weighted sum without materializing its full score/probability matrix; carry a common normalization frame for the denominator and output numerator.


## Computation before and after

Stream exponential weights with a selectable finite reference and common gain.

Work in natural scaled-logit units x. Carry a finite reference r and zero-initialized L and optionally O. With constant G=2**weight_gain_log2, the invariant is L=G*sum_seen(exp(x-r)) and O=G*sum_seen(exp(x-r)*v). For each tile choose candidate=max(r,max_finite_tile_logit); running_max selects candidate, while lagged_max keeps r when (candidate-r)/ln(2)<=threshold_log2 and selects candidate otherwise. With chosen r_new, c=exp(r-r_new). Correct exactly the old state by c, then add G*exp(x-r_new) and its weighted values for this tile. Masked elements add zero. A denominator-only extra logit a, such as a zero-valued attention sink, adds G*exp(a-r) to L in the same frame. At finalization output O/L and, when requested, natural LSE=r+log(L)-log(G), or divide that result by ln2 for base2 LSE. With finalize=none expose r,L,O,G together. Storage conversion and fast exp2 lowering are separately admitted implementations, not an exact-floating claim.

Recover the original computation: Divide L and O by their common G*exp(-r); their ratio becomes the complete normalized weighted sum, and removing log(G) recovers the original logsumexp.

## Bind the rule

1. Caller fixes score x=dot(Q,K)/sqrt(D), valid key domain, FP32 statistics and accumulation, FP16 output. Partition keys exactly once. Causal masking changes validity, never the normalization formula.
2. For the ordinary bounded replay choose state=max_sum_acc, reference_policy=running_max, threshold_log2=0, weight_gain_log2=0, rescale_point=per_tile, finalize=deferred_divide, traversal=ascending and inf_guard=true. These are choices; the causal mask and scale are caller-fixed.
3. Initialize L=O=0 with no active reference. Skip all-masked prefix tiles without evaluating an exponent or correction. The first nonempty valid tile establishes r=max_valid_score from its actual finite scaled logits; initialize L=sum(exp(x-r)) and O=sum(exp(x-r)*V) directly, without an old-state correction. An arbitrary seed such as r=0 is invalid for uniformly large negative logits: FP32 exponentials may all underflow. For each later nonempty tile choose r_new=max(r,max_valid_score), c=exp(r-r_new), then update L=c*L+sum(exp(x-r_new)) and O=c*O+sum(exp(x-r_new)*V). Invalid scores contribute zero. Keep old-state correction before adding new-frame terms.
4. With gain=1, return O/L once all valid keys are consumed and cast to FP16. Establish positive finite final mass; equal-length causal rows include their diagonal key, but padded rows must not finalize. Any probability cast to FP16 for a matrix multiply is a separate admitted numerical implementation.
5. Bind all scope booleans from this partition, initialization, mask, frame and actual arithmetic. Exact algebra does not prove a fast exponential or reduced-precision matrix multiplication meets tolerance.


| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| state | compile / software | ["max_sum", "max_sum_acc"] | Carry a reference and denominator, optionally the weighted numerator; legacy names retain max even when the selected reference can lag it. |
| rescale_point | compile / software | ["per_tile", "before_next_mma", "after_mma"] | Physical placement of correction; the implementation must correct exactly the old contributions before combining frames. |
| finalize | compile / software | ["deferred_divide", "none"] | Normalize once or expose the framed state to a later consumer. |
| traversal | compile / software | ["ascending", "descending"] | Order of a complete disjoint tile partition. |
| inf_guard | compile / software | [false, true] | Legacy implementation choice for masked-prefix handling; neither choice admits an undefined final zero denominator. |
| reference_policy | compile / software | ["running_max", "lagged_max"] | Update to the new maximum immediately or retain the old finite reference while its natural-logit increase divided by ln2 is at most threshold_log2. |
| threshold_log2 | compile / software | {"lo": 0, "hi": 8} | Maximum permitted base2 logit lag before updating the reference; source selects0 or8. |
| weight_gain_log2 | compile / software | {"lo": 0, "hi": 8} | Common weight gain G=2**weight_gain_log2 applied to denominator and numerator; source FP8 selects0 or8. |

## Coupled constraints and scope

Complete streamed positive-mass exponential reductions with finite references and a fixed positive common gain. Source FMHA uses lag8 to avoid small correction updates and FP8 gain256 at lag0 or gain1 at lag8. Nonidentity tile pruning is excluded. Source sink plus nonunit gain fails the common-frame premise until compensated; final fully masked/zero-mass rows require another binding. This is a versioned extension with an explicit finite-domain projection, not automatic promotion of every version1 source/proof or of native packed/TMEM operations.

Scope predicate: `p.assoc and p.rescalable and p.complete_disjoint_tiles and p.no_nonidentity_terms_dropped and p.initial_state_valid and p.masked_terms_are_zero and p.final_denominator_positive_finite and p.common_gain_constant and p.extra_denominator_terms_share_frame and p.numerator_denominator_reference_match and p.state_transitions_match_recurrence and p.casts_and_math_admitted and p.finalize_log_compensation_exact and p.reference_and_gain_within_numeric_range`.

- Precondition `pre_partition` (semantic): `p.assoc and p.rescalable and p.complete_disjoint_tiles and p.no_nonidentity_terms_dropped` — Every logical contribution is retained once and correction is distributive under the admitted numerical contract.
- Precondition `pre_initial` (semantic): `p.initial_state_valid and p.masked_terms_are_zero and p.final_denominator_positive_finite` — Initialization/masked prefixes are defined and final normalization has positive finite mass.
- Precondition `pre_frame` (semantic): `p.common_gain_constant and p.extra_denominator_terms_share_frame and p.numerator_denominator_reference_match` — All ordinary and extra contributions use the same reference and gain; gain is fixed across the stream.
- Precondition `pre_backend` (implementation): `p.state_transitions_match_recurrence and p.casts_and_math_admitted and p.finalize_log_compensation_exact` — Physical correction order and stored-state precision implement the declared recurrence and compensate the gain in LSE.
- Constraint `c_policy` (semantic): `reference_policy != 'running_max' or threshold_log2 == 0` — Immediate-max projection has no lag threshold.
- Constraint `c_range` (implementation): `p.reference_and_gain_within_numeric_range` — The selected reference lag and weight gain fit exponent and storage ranges including every extra denominator term.

## Local example and misuse

Local one-row recurrence: scores [0,ln(2)] and scalar values [2,4], one key per tile. Initialize r=0,L=O=0. Tile0 gives r=0,L=1,O=2. Tile1 gives r=ln(2), c=1/2,L=1.5,O=5; O/L=10/3. Multiplying the entire newly updated O by c would incorrectly rescale the new value. Shift both scores by -1000: the first tile establishes r=-1000, then the second advances r to -1000+ln(2). L, O and output are unchanged within FP32 rounding. Seeding r=0 instead would underflow both weights and create a zero denominator. For a masked prefix, preserve the uninitialized reference and zero state until the first valid tile. These are scalar invariant witnesses, not an attention implementation.


## Implementations, evidence and extension boundary

The unchanged CUTLASS study contract also exposes lagged references and weight gains; they are optional refinements, not required for FP16 attention. The simplest binding uses running_max and gain1. Source FMHA witnesses do not certify a TileLang implementation or every source-menu point. Exp2 lowering must transform scaled logits consistently by log2(e); never mix natural scores with an unconverted base2 exponent.


Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/online_rescaled_reduction@2/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/online_rescaled_reduction@2/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Saves full S-by-S score/probability traffic at the cost of per-tile maxima, exponentials and accumulator correction. Larger key tiles reduce update frequency but increase working state. Large query tiles increase the live output numerator. Underflow, score range, cancellation in V and probability rounding need measured numerical acceptance.


```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `invariant` (proof): After every tile the state equals the retained prefix sums in the declared reference/gain frame; reference need not equal the exact maximum.
- `finalize_equivalence` (test): Common-frame normalization and gain-compensated LSE recover the original positive-mass operator under the declared numerical policy.
- `extra_terms` (test): Every denominator-only contribution receives the same reference and gain as ordinary weights.
- `masked_prefixes` (test): Empty masked prefixes have defined state transitions and final zero-mass rows are excluded or handled by a separate admitted variant.
- `implementation_order` (test): Correction, cast, exp/log and storage ranges realize the recurrence without applying an old-frame factor to new-frame contributions.

Finite real identities do not admit overflowing exponentials, P quantization, fused arithmetic or dropped mass without their declared numerical policies.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- enables: approximate_math_rewrite (outside menu); Exp2 with folded scale implements the same frame when its numerical policy is admitted.

Structured contract: [template.yaml](template.yaml).
