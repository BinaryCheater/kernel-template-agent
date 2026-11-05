# hierarchical_centroid_budget_seeding@1: hierarchical centroid budget seeding

## When to consider

When training many centers is expensive, allocate a fixed fine-center budget across coarse partitions, train each partition separately and use the concatenated centers to seed a global refinement. Populations[8,0,2] and fine budget5 allocate[4,0,1] with prefix[0,4,4,5]. Independently rounding fractions without reserving remaining nonempty groups can exhaust the budget early.

## Computation before and after

When training many centers is expensive, allocate a fixed fine-center budget across coarse partitions, train each partition separately and use the concatenated centers to seed a global refinement.

Train coarse groups, count their populations and allocate fine-center quotas proportional to remaining population and remaining budget, reserving at least one center for each remaining nonempty group. Prefix-sum quotas to assign disjoint fine-center output intervals. Gather each group training rows, optionally cap its training population as an explicit quality approximation, train its local centers and write into its assigned interval. Concatenate exactly K centers, then refine globally under the declared iteration policy. Empty coarse groups receive zero quota; no partition-local assignment is mistaken for final global nearest-center membership.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| coarse_centers | compile / software | {"lo": 1, "hi": 65536} | Number of coarse mesoclusters; source chooses min(K,round(sqrt(K))) as a heuristic. |

## Coupled constraints and scope

NVIDIA hierarchical balanced KMeans center initialization. Source rounds quota sequentially with remaining budgets, so naïvely rounding independent proportional quotas is not equivalent. Does not guarantee exact Lloyd optimum or strict population balance. Resource effects: Reduces all-center training into smaller local problems plus coarse training and global refinement; adds gather buffers and quota metadata. Resource bound depends on largest retained training group.

Scope predicate: `p.coarse_sizes_partition_rows and p.fine_budget_feasible and p.quota_sum_exact_and_offsets_disjoint and p.training_samples_available and p.global_refinement_contract_defined`.

- Precondition `pre_0` (semantic): `p.coarse_sizes_partition_rows and p.fine_budget_feasible` — Coarse populations sum to all rows and total fine budget covers every nonempty coarse group.
- Precondition `pre_1` (semantic): `p.quota_sum_exact_and_offsets_disjoint and p.training_samples_available` — Nonempty quotas are positive, quotas sum to requested K, their prefix intervals are disjoint, and selected train samples cover each requested local center count.
- Precondition `pre_2` (semantic): `p.global_refinement_contract_defined` — Local center concatenation is initialization; optional global refinement and final assignment semantics are explicit.

## Local example and misuse

For a local instance of `map`, identify the object `hierarchical_centroid_budget_seeding` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cuvs/delta/hierarchical_centroid_budget_seeding@1/template.yaml`. Original outline: `knowledge/studies/cuvs/delta/hierarchical_centroid_budget_seeding@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Reduces all-center training into smaller local problems plus coarse training and global refinement; adds gather buffers and quota metadata. Resource bound depends on largest retained training group.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `budget` (proof): Every nonempty group receives a positive quota and all quotas sum exactly to K.
- `output_ownership` (proof): Each local training group writes only its assigned center interval.
- `sample_scope` (proof): Training caps and subsampling are recorded as approximation choices, not coverage of all rows.
- `refinement_boundary` (proof): Global prediction/refinement consumes the complete center seed only after every local trainer finishes.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
