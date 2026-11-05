# quantization_candidate_error_selection@1: quantized candidate choice

## When to consider

When two admitted low-bit scale/code candidates trade clipping against resolution, construct both full stored representations and select using a declared reconstruction-error score over the entire scale-sharing domain. Store the winning codes and scale together. This guarantees the smaller computed score, not optimality among every possible scale.

## Computation before and after

Evaluate two stored low-float representations and select their lower complete-group error.

For coarse amax M and local absmax a, construct S_enc from outer_scale_max*6/M. Let b=a/6*S_enc. Stored E4M3 candidates are s6=cast(min(b,448)) and s4=cast(min(1.5*b,448)). Encode all values twice into E2M1 using the multiplier from the corresponding stored scale and common coarse inverse. Score each complete candidate by summing the requested error metric. fp32_decoded uses the native decoded product q*s*M/(6*outer_scale_max) under explicit FP32 round operations; fp16_scaled_proxy compares rounded FP16 q*s against x*S_enc. For square16 reduce errors over all sixteen row groups before choosing, so all shared scales make one decision. Choose map4 only if score4<score6, otherwise map6. Emit the chosen codes and chosen stored scale as one representation. Candidate clipping/scale casts can make nominal map4/map6 spacing differ from ideal real arithmetic.

Recover the original computation: Select the corresponding representation from an explicit two-candidate program; the output is one admitted candidate with the declared smaller computed score.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| metric | compile / boundary | ["absolute", "squared"] | Requested aggregate absolute-error or squared-error score; division by common group size is unnecessary for comparison. |
| score_arithmetic | compile / software | ["fp32_decoded", "fp16_scaled_proxy"] | Source offers FP32 decode-domain scoring or a faster encoded-domain score with FP16 products; the latter is a different score policy and can change the winner. |
| group_shape | compile / boundary | ["vector16", "square16"] | One sixteen-value group or shared sixteen-by-sixteen group whose errors must be summed before choosing. |
| outer_scale_max | compile / boundary | [256, 448] | Coarse E4M3 range reservation used to construct both map4/map6 candidates. |

## Coupled constraints and scope

Pinned NVFP4 map4/map6 candidates, E4M3 stored scales, E2M1 codes, compact scale metadata and no stochastic rounding. Source rowwise implementation supports vector16 or square16 with complete dimensions; row-scaled outer maxima cannot use square16 or columnwise output. This logical rule does not supply a native converter, proxy-to-true-error guarantee or a search over all scales. IEEE exceptional values and ties in nonfinite scores are excluded.

Scope predicate: `p.cols % 16 == 0 and (group_shape != 'square16' or p.rows % 16 == 0) and (score_arithmetic != 'fp16_scaled_proxy' or p.proxy_metric_admitted) and p.error_selected_nvfp4_recipe_requested and p.deterministic_rounding and p.positive_finite_scale_domain and p.same_inputs_and_group_domain and p.codes_generated_from_stored_scales and p.both_candidate_scores_finite and p.full_group_error_reduced and p.score_arithmetic_matches and p.winning_codes_and_scale_paired`.

- Precondition `pre_recipe` (semantic): `p.error_selected_nvfp4_recipe_requested and p.deterministic_rounding and p.positive_finite_scale_domain` — The requested NVFP4 recipe admits a choice between map4/map6 with deterministic conversions and finite positive scale factors.
- Precondition `pre_candidates` (semantic): `p.same_inputs_and_group_domain and p.codes_generated_from_stored_scales and p.both_candidate_scores_finite` — Both candidates encode the same complete logical values using their actual stored scales.
- Precondition `pre_select` (implementation): `p.full_group_error_reduced and p.score_arithmetic_matches and p.winning_codes_and_scale_paired` — Group error and selection use the specified arithmetic and pair the winning codes and metadata.
- Constraint `c_groups` (semantic): `p.cols % 16 == 0 and (group_shape != 'square16' or p.rows % 16 == 0)` — Candidate groups are complete.
- Constraint `c_proxy` (semantic): `score_arithmetic != 'fp16_scaled_proxy' or p.proxy_metric_admitted` — A faster score may be selected only when the requested policy admits it.

## Local example and misuse

For a local instance of `contract`, identify the object `quantized_candidate_choice` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/transformerengine/delta/quantization_candidate_error_selection@1/template.yaml`. Original outline: `knowledge/studies/transformerengine/delta/quantization_candidate_error_selection@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Two candidate code packets and scores until selecting one code/scale pair.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `candidate_identity` (test): Scored decoded values come from the same codes and stored scale eventually emitted.
- `group_score` (proof): All values sharing the selected scale participate in one completed error reduction.
- `selection` (proof): The selected candidate has score no larger than the other under the declared arithmetic and tie rule.

FP16 encoded-domain proxy scoring can rank candidates differently from FP32 decoded-domain scoring; a lower computed score does not prove lower error for arbitrary downstream GEMMs.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
