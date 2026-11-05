# keyed_weighted_race_selection@2: deterministic weighted token race

## When to consider

When weighted token selection repeats under changed launch partition, bind noise to original token identity and reduce identical score/ID records. Choose direct log(U)/weight or an admitted log-weight Gumbel score to avoid explicit normalization; retain random orientation, counter mapping and dtype steps.

## Computation before and after

Partition-invariant keyed weighted race with direct linear or unnormalized log-weight score encoding.

For each supported original token i, produce a32-bit word h from the fixed key/context/token identity. Let u0=(h+.5)/2^32 and set U=u0 for midpoint or U=1-u0 for complement_midpoint. For linear_log_uniform compare log(U)/w_i. For log_weight_gumbel compare logw_i-log(-log(U)); a common normalization of weights subtracts the same constant and is unnecessary. Evaluate log(U) with a declared endpoint-safe recipe: complement_midpoint uses log1p(-u0); midpoint may use the complement near1 as in version1. Reduce each complete shard and its summaries by larger score then smaller original token ID. Counter word lanes and global vocabulary origins remain fixed across sharding. Context_mixture still chooses one key for the entire context. Fixed input casts/bias/temperature and optional stored logits remain part of the selected workload, not removable rounding.

Recover the original computation: For a fixed score recipe and original token/random identity, replace shard summaries by their records; associative lexicographic max preserves the token. In exact arithmetic logw-log(-log(U))=-log(-log(U)/w) is strictly increasing in log(U)/w, so the two score forms have the same argmax when U and weights correspond. Finite native score-form changes require their admitted rounding policy.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| split_width | compile / software | [1024, 8192] | Token/rank entries per independently reduced shard. |
| key_mode | compile / boundary | ["single", "context_mixture"] | Single key or a context-wide dual-key decision; the requested selector identity fixes this mode. |
| score_form | compile / software | ["linear_log_uniform", "log_weight_gumbel"] | Compare log(U)/w directly, or compare log(w)-log(-log(U)); the latter can consume unnormalized log weights without materializing softmax. |
| uniform_map | compile / boundary | ["midpoint", "complement_midpoint"] | Underlying U is (h+.5)/2^32 or1-(h+.5)/2^32. This records requested random coupling; changing it changes the fixed-hash outcome. |

## Coupled constraints and scope

Version1 TextSeal split8192 midpoint linear score binding is unchanged. Version2 adds DSpark Markov block1024, log_weight_gumbel,complement_midpoint and single key with10-round Philox words identified by global vocabulary quad,request row,step and offset. Its global vocabulary shard starts must be4-aligned to preserve the declared word-to-token map. Input bias is added in FP32 then rounded to the declared step dtype before temperature/noise. Native u may round to1 and create an infinite exponential; finite-hash outcome uses the declared endpoint policy. No exact categorical law or independence is inferred from a finite hash/counter generator.

Scope predicate: `(p.support_nonempty and p.inactive_entries_excluded and ((score_form == 'linear_log_uniform' and p.active_weights_finite_positive) or (score_form == 'log_weight_gumbel' and p.active_logweights_finite))) and (p.hash_recipe_fixed_by_key_context_token and p.hash_independent_of_shard_and_traversal) and (key_mode != 'context_mixture' or p.one_key_choice_shared_by_entire_context) and (p.selected_score_recipe_and_uniform_endpoint_policy_admitted and p.score_ties_choose_smallest_original_token_id) and (p.shards_partition_complete_support and p.local_and_global_comparator_identical)`.

- Precondition `pre_support` (semantic): `p.support_nonempty and p.inactive_entries_excluded and ((score_form == 'linear_log_uniform' and p.active_weights_finite_positive) or (score_form == 'log_weight_gumbel' and p.active_logweights_finite))` — Supported records use positive finite weights or finite log weights; masks and nonfinite endpoint fallback have an explicit policy.
- Precondition `pre_identity` (semantic): `p.hash_recipe_fixed_by_key_context_token and p.hash_independent_of_shard_and_traversal` — Each original token gets the same 32-bit hash regardless of launch partition.
- Precondition `pre_mixture` (semantic): `key_mode != 'context_mixture' or p.one_key_choice_shared_by_entire_context` — Dual-key mixture chooses one key for a context, not a separate key for each token.
- Precondition `pre_score` (semantic): `p.selected_score_recipe_and_uniform_endpoint_policy_admitted and p.score_ties_choose_smallest_original_token_id` — Compute the selected score from the exact declared hash/counter identity and orientation, with matching native rounding/endpoint and score/ID comparison policy. Version1 log/divide admission implies this premise for its unchanged binding.
- Precondition `pre_reduce` (semantic): `p.shards_partition_complete_support and p.local_and_global_comparator_identical` — No support entry omitted or duplicated; each summary retains both score and original token ID.

## Local example and misuse

For a local instance of `reduce`, identify the object `deterministic_weighted_token_race` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/keyed_weighted_race_selection@2/template.yaml`. Original outline: `knowledge/studies/sglang/delta/keyed_weighted_race_selection@2/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Per-shard summaries and original row inputs.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `partition_invariance` (proof): Every launch partition yields the same token for identical per-token scores and support.
- `identity` (proof): Hash inputs use original token identity; reordering rank positions does not change scores.
- `numeric_endpoint` (test): Native conversion/log1p/division retains the admitted finite endpoint and tie policy.

Exact monotonic score equivalence does not imply identical FP32 ties/ranks. Preserve uniform orientation, counter-lane mapping, step casts, temperature and endpoint policy. Finite32-bit uniforms may round to1; generator quality and ideal continuous sampling law are separate premises.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
