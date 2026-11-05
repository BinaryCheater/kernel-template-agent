# keyed_weighted_race_selection@1: deterministic weighted token race

## When to consider

When weighted token selection must repeat for the same key and context despite changed launch partition, derive each race score from original token identity, then reduce shard winners with the same score/ID order.

## Computation before and after

When weighted token selection must repeat for the same key and context despite changed launch partition, derive each race score from original token identity, then reduce shard winners with the same score/ID order.

For each supported token i, compute h=H(key,context,i), u=(h+0.5)/2^32 and score=log(u)/weight_i. Preserve strict negative log near u=1 by evaluating log1p(-((2^32-1-h)+0.5)/2^32) in the upper half. Reduce each shard by larger score, then smaller original token ID; reduce those summaries with the same comparator. In context_mixture mode hash the two keys and context once to choose the key shared by all tokens. Support filters (top-k, top-p, min-p) are fixed before this race. Logit clearing and one-hot forcing occur after the selected token is determined.

Recover the original computation: Replace the two-stage comparator reduction by a single reduction over the same keyed score records; associative lexicographic max preserves the selected token.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| split_width | compile / software | [8192] | Token/rank entries per independently reduced shard. |
| key_mode | compile / boundary | ["single", "context_mixture"] | Single key or a context-wide dual-key decision; the requested selector identity fixes this mode. |

## Coupled constraints and scope

TextSeal source ops/sampling/textseal_selector.py:33-40,297-377,786-907; split8192. The direct-token path uses leftmost ties because contiguous token-ID shards are ascending; sorted-rank path explicitly reduces token IDs. Empty/all-infinite scores need a separate caller policy. This is equivalence to the fixed hash selector; a categorical-law claim additionally requires independent continuous uniforms and is not implied by a finite Murmur hash.

Scope predicate: `(p.support_nonempty and p.active_weights_finite_positive and p.inactive_entries_excluded) and (p.hash_recipe_fixed_by_key_context_token and p.hash_independent_of_shard_and_traversal) and (key_mode != 'context_mixture' or p.one_key_choice_shared_by_entire_context) and (p.same_admitted_log_divide_recipe_for_all_scores and p.score_ties_choose_smallest_original_token_id) and (p.shards_partition_complete_support and p.local_and_global_comparator_identical)`.

- Precondition `pre_support` (semantic): `p.support_nonempty and p.active_weights_finite_positive and p.inactive_entries_excluded` — Only positive finite admitted weights compete; at least one finite score exists.
- Precondition `pre_identity` (semantic): `p.hash_recipe_fixed_by_key_context_token and p.hash_independent_of_shard_and_traversal` — Each original token gets the same 32-bit hash regardless of launch partition.
- Precondition `pre_mixture` (semantic): `key_mode != 'context_mixture' or p.one_key_choice_shared_by_entire_context` — Dual-key mixture chooses one key for a context, not a separate key for each token.
- Precondition `pre_score` (semantic): `p.same_admitted_log_divide_recipe_for_all_scores and p.score_ties_choose_smallest_original_token_id` — Map h to u=(h+0.5)/2^32, evaluate log(u)/weight by the declared endpoint-safe recipe; compare score then token identity.
- Precondition `pre_reduce` (semantic): `p.shards_partition_complete_support and p.local_and_global_comparator_identical` — No support entry omitted or duplicated; each summary retains both score and original token ID.

## Local example and misuse

For a local instance of `reduce`, identify the object `deterministic_weighted_token_race` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/keyed_weighted_race_selection@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/keyed_weighted_race_selection@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

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

Lexicographic maximum is associative and each shard winner dominates every record it replaces. Complement log1p avoids rounding the largest 32-bit hash to u=1. This proof preserves a declared finite-hash selector, not cryptographic quality or an exact categorical law. Native underflow/division may create score ties or infinities; source requires a matching policy. Host partition/endpoint witnesses are checks/keyed_race.json.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
