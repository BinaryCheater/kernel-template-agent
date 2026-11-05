# bounded_normalized_weight_pruning@1: normalized weight pruning certificate

## When to consider

When small-score attention tiles are candidates for omission, use their maxima only to propose skips. Accumulate a discarded-mass bound in the same frame as retained mass, include value ranges and enforce absolute output and optional LSE budgets. Keep collective protocol retirement independent of skipped arithmetic. Source threshold heuristics without those certificates remain conditional instances.

## Computation before and after

Certify normalized weighted error before accepting nonidentity tile pruning.

For each output row let kept nonnegative weights sum to K>0 and dropped weights sum to D>=0. Maintain conservative K_lower<=K and B>=D in one positive common weight frame. A tile with n live terms and maximum scaled logit u at reference r contributes at most n*exp(u-r) to B; if exp(u-r)<ratio_threshold then n*threshold is also a valid strict upper bound. Accumulate all dropped-tile bounds, not just the latest tile. When r or common gain changes, rescale B and K_lower identically. Propose skips collectively but admit the completed dropped set only if output_abs_scale*value_diameter*B/(K_lower+B) plus additional output error fits output_error_budget. If LSE is requested certify log1p(B/K_lower) plus its arithmetic error in natural-log units. Otherwise retain the proposed terms or choose a different kernel. Compute the normalized output over kept terms; neither the unnormalized denominator nor gradients are claimed unchanged.

Recover the original computation: Reinsert the dropped terms and use the convex-combination identity between kept and dropped weighted means to bound the complete operator's output and log-normalizer difference.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| ratio_threshold | runtime / software | {"lo": 0, "hi": 1} | Propose a tile skip when its maximum weight relative to the current reference is below this positive threshold; a certificate is still required. |
| output_error_budget | runtime / boundary | {"lo": 0, "hi": 1000000000} | Allowed absolute infinity-norm output error including additional admitted arithmetic and conversion error. |
| require_lse | compile / boundary | [false, true] | Whether natural-log normalizer error also has to satisfy the declared budget. |
| lse_error_budget | runtime / boundary | {"lo": 0, "hi": 16} | Absolute natural-log normalizer error budget; convert explicitly if the interface reports base2 LSE. |

## Coupled constraints and scope

Normalized finite nonnegative weighted sums with positive retained mass and an absolute output-error contract; optional LSE budget is in natural units. Source FMHA supplies a collective maximum-ratio proposal but no accumulated mass/value range certificate, so its threshold alone is not an admitted implementation of this rule. Finite source examples can be certified separately; generic native skipping, reference parity with the host, accumulator initialization and backward require further evidence. Relative error, signed weights, zero kept mass and unchanged gradients are excluded.

Scope predicate: `p.nonnegative_finite_weights and p.kept_dropped_partition_exact and p.certificate_covers_all_skips and p.kept_mass_lower_valid and p.dropped_mass_upper_valid and p.coordinate_value_ranges_valid and p.bounds_outward_rounded and p.all_mass_bounds_share_frame and p.extra_denominator_terms_accounted and p.kept_output_is_normalized and p.absolute_error_contract and p.additional_error_bounds_valid and p.lse_bound_valid and p.collective_skip_uniform and p.skipped_work_protocol_retired and p.initial_kept_state_valid`.

- Precondition `pre_partition` (semantic): `p.nonnegative_finite_weights and p.kept_dropped_partition_exact and p.certificate_covers_all_skips` — Every ordinary or extra term is kept or dropped once and the accumulated bound covers every dropped term.
- Precondition `pre_bounds` (semantic): `p.kept_mass_lower_valid and p.dropped_mass_upper_valid and p.coordinate_value_ranges_valid and p.bounds_outward_rounded` — Retained mass lower bound and dropped mass upper bound are conservative; all value coordinates lie in intervals of at most the declared diameter.
- Precondition `pre_frame` (semantic): `p.all_mass_bounds_share_frame and p.extra_denominator_terms_accounted and p.kept_output_is_normalized` — Rescale retained mass and discarded-mass bounds together; denominator-only terms use zero values included in the ranges.
- Precondition `pre_contract` (semantic): `p.absolute_error_contract and p.additional_error_bounds_valid and p.lse_bound_valid` — The contract requests these absolute bounds; LSE upper bound is at least log1p(B/K_lower) plus admitted LSE arithmetic error when requested.
- Precondition `pre_protocol` (implementation): `p.collective_skip_uniform and p.skipped_work_protocol_retired and p.initial_kept_state_valid` — One collective decision applies to its whole work unit and bypassed arithmetic still retires the right pipeline generation without reading uninitialized accumulators.
- Constraint `c_positive` (semantic): `ratio_threshold > 0 and p.kept_mass_lower > 0 and p.dropped_mass_upper >= 0 and p.value_diameter >= 0 and p.output_abs_scale >= 0 and p.additional_output_error_upper >= 0` — Certificate masses and error allowances are valid with positive retained mass.
- Constraint `c_output` (semantic): `p.output_abs_scale * p.value_diameter * p.dropped_mass_upper / (p.kept_mass_lower + p.dropped_mass_upper) + p.additional_output_error_upper <= output_error_budget` — Worst-case normalized output movement plus arithmetic error fits the absolute budget.
- Constraint `c_lse` (semantic): `not require_lse or (p.lse_error_upper >= 0 and p.lse_error_upper <= lse_error_budget)` — A certified natural-log bound including additional LSE arithmetic error fits the requested budget.

## Local example and misuse

For a local instance of `reduce`, identify the object `normalized_weight_pruning_certificate` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/bounded_normalized_weight_pruning@1/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/bounded_normalized_weight_pruning@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Per-row conservative retained/dropped mass and value-range bounds.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `mass_certificate` (exhaustive): The bounds conservatively cover the complete kept/dropped partition through every reference-frame change.
- `output_bound` (proof): Each complete mean differs from the kept mean by at most diameter*D/(K+D) before output scaling and additional numerical error.
- `lse_bound` (test): The complete natural LSE exceeds kept LSE by log1p(D/K) and this is bounded by the certificate.
- `collective_protocol` (test): Every row in a skipped collective satisfies its certificate and bypassing arithmetic preserves storage initialization and pipeline ownership.

A small maximum individual weight alone gives no global mass or output bound; rounding of certificates and additional native numerical error must be conservative.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- enables: [online_rescaled_reduction@2](../../online_rescaled_reduction/2/GUIDE.md); The certified kept operator can be streamed in a common frame; any frame change must also correct its discarded-mass certificate.

Structured contract: [template.yaml](template.yaml).
