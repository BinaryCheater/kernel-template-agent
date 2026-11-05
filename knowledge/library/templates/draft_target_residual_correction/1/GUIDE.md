# draft_target_residual_correction@1: draft chain target law coupling

## When to consider

When a fast draft proposes a chain and target probabilities for those histories are available, reuse its consecutive prefix with target/draft acceptance. At first rejection draw one token from positive target-minus-draft mass; after full acceptance draw the target bonus. Keep actual emitted-prefix length separate from independent diagnostic acceptance counts, preserve conditional RNG identity and handle finite endpoints/zero residual explicitly.

## Computation before and after

Draft-prefix acceptance plus one positive target-minus-draft correction, or a target bonus after full acceptance.

Use Q for draft and P for target probabilities, regardless of source variable spelling. For t=0..L-1, draw an independent acceptance uniform and accept proposed d_t with probability min(1,P_t[d_t]/Q_t[d_t]); emit it only while every earlier draft was accepted. Stop at first rejection r. If r<L, compute the complete nonnegative residual weights max(P_r-Q_r,0), reduce their mass and draw exactly one correction token from that normalized residual; store it at r. If all L pass, draw one bonus token from P_L and store at L. Mark all remaining output slots invalid. emitted_draft_count=r. In resample_unemitted mode, additionally recheck positions r..L-1 using fresh independent acceptance draws and accumulate diagnostic_count=r+number of passing rechecks; include r again and do not change the output prefix. Preserve additive diagnostic/emitted counter deltas and their exclusive writers. The source sequential RNG consumes L+2 uniform draws on rejection and L+1 when all drafts pass; removing diagnostics changes final-draw position unless an explicit counter map supplies the required identity.

Recover the original computation: For one position, accepted joint mass of token i is min(Q_i,P_i); reject mass is sum_i(P_i-Q_i)_+, whose correction adds(P_i-Q_i)_+, giving P_i. Apply this coupling at successive matched histories; after correction continue from the target at the corrected history. The variable-length emitted block is a stopping prefix of a target-law continuation, not a claim that its distribution conditioned on block length is the target prefix law. Diagnostic rechecks affect counters/random-stream positions only.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| diagnostic_mode | compile / boundary | ["none", "resample_unemitted"] | Requested observability: emitted prefix only, or source-style independently rechecked acceptance count for all unemitted positions. Source uses resample_unemitted; none is inferred and needs an explicit RNG identity policy. |

## Coupled constraints and scope

Source sampling.cuh:ChainSpeculativeSampling handles one chain/CTA with full draft/target rows and a final target bonus row, then pads remaining slots-1. Source calls target probability q and draft probability p; this candidate uses conventional P/Q labels. Diagnostic rechecks include the rejected position. Literal strict u*Q<P with CURAND including u1 can reject equal rows and leave zero residual; the native endpoint/zero-mass premise is therefore a named source boundary rather than an established exact-law realization. No speculative tree, arbitrary proposal generator or finite-grid exactness is implied.

Scope predicate: `p.vocabulary >= 1 and p.draft_length >= 1 and p.output_capacity >= p.draft_length + 1 and (p.draft_and_target_rows_finite_nonnegative_normalized and p.actual_draft_tokens_drawn_from_declared_conditional_draft_rows) and (p.target_and_draft_rows_match_same_generation_and_accepted_history and p.target_bonus_row_available) and (p.fresh_acceptance_and_correction_randomness_with_explicit_invocation_position_components) and (p.accept_with_min_one_target_over_draft_policy and p.native_endpoint_and_zero_residual_policy_match_contract) and (p.correction_is_positive_target_minus_draft_and_uses_its_complete_mass and p.zero_mass_correction_unreachable_or_explicitly_handled) and (p.only_consecutive_accepted_drafts_are_emitted and p.one_correction_or_bonus_then_invalid_tail) and (diagnostic_mode != 'resample_unemitted' or (p.diagnostic_rechecks_use_separate_draws_and_never_extend_output_prefix and p.counter_deltas_and_exclusive_updates_explicit)) and (p.acceptance_products_residual_subtraction_and_cdf_rounding_admitted and p.output_generation_published_before_readers)`.

- Precondition `pre_distributions` (implementation): `p.draft_and_target_rows_finite_nonnegative_normalized and p.actual_draft_tokens_drawn_from_declared_conditional_draft_rows` — Q_t and P_t are normalized distributions at the same history; every proposed draft token was actually generated from that Q_t. A token with Q_t(token)=0 cannot be proposed under this premise.
- Precondition `pre_history` (implementation): `p.target_and_draft_rows_match_same_generation_and_accepted_history and p.target_bonus_row_available` — At position t both rows condition on the preceding drafted prefix, which equals the emitted prefix while all earlier drafts were accepted; an extra target row exists after the full draft chain.
- Precondition `pre_rng` (implementation): `p.fresh_acceptance_and_correction_randomness_with_explicit_invocation_position_components` — Acceptance draws and final categorical correction/bonus draws have the declared independent conditional law and explicit identity. Diagnostic draws are distinct and their effect on sequential RNG offsets is retained.
- Precondition `pre_acceptance` (implementation): `p.accept_with_min_one_target_over_draft_policy and p.native_endpoint_and_zero_residual_policy_match_contract` — Accept a draft token with probability min(1,P(token)/Q(token)), implemented without unsafe division when possible. Equal distributions cannot enter a positive-probability reject branch with zero correction mass; native endpoints require an explicit matching policy.
- Precondition `pre_residual` (implementation): `p.correction_is_positive_target_minus_draft_and_uses_its_complete_mass and p.zero_mass_correction_unreachable_or_explicitly_handled` — At the first rejection sample R_i=max(P_i-Q_i,0)/sum_j max(P_j-Q_j,0), not from P again or from Q-P. The full correction domain/mass is complete; no undefined zero-mass CDF occurs.
- Precondition `pre_prefix` (implementation): `p.only_consecutive_accepted_drafts_are_emitted and p.one_correction_or_bonus_then_invalid_tail` — Emit only the accepted prefix, then one corrected token at first rejection or one P_L bonus when all drafts pass; later positions remain invalid, not additional independently accepted drafts.
- Precondition `pre_diagnostics` (implementation): `diagnostic_mode != 'resample_unemitted' or (p.diagnostic_rechecks_use_separate_draws_and_never_extend_output_prefix and p.counter_deltas_and_exclusive_updates_explicit)` — Optional source-style diagnostic count adds fresh independent acceptance rechecks for every unemitted draft, including the first rejected position. It can exceed the emitted prefix; counter updates retain additive/exclusive state semantics.
- Precondition `pre_numeric` (implementation): `p.acceptance_products_residual_subtraction_and_cdf_rounding_admitted and p.output_generation_published_before_readers` — Native comparison/product, positive subtraction, sum and CDF endpoint policies satisfy the probability/error contract; one complete output generation publishes before consumers.
- Constraint `c_chain` (semantic): `p.vocabulary >= 1 and p.draft_length >= 1 and p.output_capacity >= p.draft_length + 1` — A finite nonempty draft chain, a target bonus row and output/invalid-tail capacity.

## Local example and misuse

For a local instance of `contract`, identify the object `draft_chain_target_law_coupling` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/draft_target_residual_correction@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/draft_target_residual_correction@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Draft/target conditional rows, accepted prefix and L+1 output slots; optional additive diagnostic counters.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `single_step_identity` (proof): Accepted mass plus rejected positive-residual mass equals the target probability at every vocabulary coordinate, including disjoint supports.
- `chain_history` (test): All accepted prefixes use matching Q/P histories and the bonus/corrected continuation has the target joint law.
- `prefix_and_diagnostics` (test): Only consecutive accepted drafts are emitted; diagnostic rechecks use distinct draws and cannot lengthen that prefix or replace the correction.
- `native_endpoint_and_state` (test): Finite acceptance endpoints, zero correction mass, CDF fallback, RNG positions, counter deltas and output publication implement the admitted native policy.

Acceptance endpoints and FP32 subtraction can create a zero residual in a reachable native reject branch. Incorrect proposal provenance, reversed residual, fresh-target correction or diagnostic/output conflation changes the distribution.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
