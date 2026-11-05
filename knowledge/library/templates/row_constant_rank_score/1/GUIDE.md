# row_constant_rank_score@1: row constant rank score

## When to consider

When all candidates in a query share a metric term, rank by a cheaper score and reconstruct the reported metric only for winners under an explicit order/tie contract. For one query norm25, candidates scores-10,-9 rank in that order and reconstruct L2 distances5,7. Clamping two negative rounded reconstructions to0 can create a tie that did not exist in score space; do not substitute final-float smallest-ID tie semantics.

## Computation before and after

When all candidates in a query share a metric term, rank by a cheaper score and reconstruct the reported metric only for winners under an explicit order/tie contract.

For squared L2, score=0.5*database_norm_squared-query_dot_database; select minimum score, then report query_norm_squared+2*score, optionally nonnegative clamp and square root. For cosine, rank by -dot/database_norm for positive database norms and reconstruct1+score/query_norm after selecting; handle zero query/database norms by the declared convention. Compute per-query constants once and omit them from the candidate inner loop. Candidate values and identity are recovered together only for selected winners.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

NVIDIA cuTile fused1NN and Faiss L2 selection. Source cuTile cosine zero norm returns1, while IVF-SQ cosine zero norm returns0; those are different contracts. Float TF32 conversion and norm generation must agree; exact algebra alone does not prove identical floating winner identities. Resource effects: Avoids repeating query constants and full metric epilogues over every candidate; retains one score/ID pair per query, with norm reads and reconstruction only for winners.

Scope predicate: `p.score_transform_monotone and p.query_constants_valid and p.rounding_and_tie_policy_admitted and p.winner_identity_preserved`.

- Precondition `pre_0` (semantic): `p.score_transform_monotone and p.query_constants_valid` — For a fixed query, the declared positive-scale/constant transformation preserves the required candidate order; any norm denominator is valid or has explicit zero policy.
- Precondition `pre_1` (semantic): `p.rounding_and_tie_policy_admitted` — Caller admits ranking arithmetic before reconstruction; rounding, clamps and nonlinear transforms do not silently impose a stronger final-value tie rule.
- Precondition `pre_2` (semantic): `p.winner_identity_preserved` — Selected score and candidate ID remain paired through reconstruction.

## Local example and misuse

For a local instance of `map`, identify the object `row_constant_rank_score` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cuvs/delta/row_constant_rank_score@1/template.yaml`. Original outline: `knowledge/studies/cuvs/delta/row_constant_rank_score@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Avoids repeating query constants and full metric epilogues over every candidate; retains one score/ID pair per query, with norm reads and reconstruction only for winners.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `order` (proof): The score order maps to the requested metric ranking for every admitted candidate, including zero norms.
- `numeric_boundary` (proof): Any pre/post clamp or rounding-induced ties follow the declared transformed-score contract, not an assumed equivalent stable sort of reconstructed floats.
- `constants` (proof): Query/database norms match the same operand representation and MMA rounding policy as the dot products.
- `reconstruction` (proof): Returned metric uses the winner original score and correct query constant exactly once.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
