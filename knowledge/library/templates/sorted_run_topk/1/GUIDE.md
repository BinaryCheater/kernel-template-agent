# sorted_run_topk@1: sorted candidate runs

## When to consider

Use for exact top-k when candidates are distributed in small local groups. Choose run_length, bind the requested k, sort each run once and consume only the winning head each round. Runs [(7,id3),(2,id0)] and [(7,id1),(4,id2)] yield ids1,3,2 for k3. Keep descending-score/smallest-id order identical in local sorting and global merging.

## Computation before and after

Sort local runs and repeatedly merge their best unconsumed heads.

Sort each run by descending score then ascending candidate identity. Start every cursor at zero. For each of k rounds offer the first unconsumed candidate of every nonempty run, select the globally best head under the same total order, emit that identity and advance only its run's cursor. Exhausted runs offer no candidate.

Recover the original computation: Forget the runs and cursors; emitted identities are the first k entries of a global sort under the requested order.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| run_length | compile / software | {"lo": 1, "hi": 32} | Maximum number of candidates stored and sorted together in each run. |
| k | compile / boundary | {"lo": 1, "hi": 32} | Requested output count; this changes the workload rather than its numerical approximation. |

## Coupled constraints and scope

Exact finite-score top-k selection with distinct identities and deterministic smallest-identity tie breaking. Runs may be shorter than their capacity. NaN ranking, approximate selection, duplicate identities and changed score transformations are outside this rule. Run count is derived from the chosen partition, not fixed here.

Scope predicate: `p.finite_valid_scores and p.unique_candidate_ids and p.disjoint_complete_runs`.

- Precondition `pre_order` (semantic): `p.finite_valid_scores and p.unique_candidate_ids and p.tie_break_smallest_id` — Descending score then ascending identity is a total order on valid candidates.
- Precondition `pre_mask` (semantic): `p.invalid_candidates_excluded and p.comparisons_preserve_order` — Padding and exhausted runs never win and comparisons implement the requested order.
- Constraint `c_requested` (semantic): `k == p.requested_k and k <= p.valid_count` — Return exactly the requested number of valid candidates.
- Constraint `c_runs` (semantic): `p.run_capacity == run_length and p.disjoint_complete_runs` — Runs fit their storage and partition the valid candidate identities.

## Local example and misuse

For a local instance of `reduce`, identify the object `sorted_candidate_runs` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/deepgemm/delta/sorted_run_topk@1/template.yaml`. Original outline: `knowledge/studies/deepgemm/delta/sorted_run_topk@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Candidate runs plus one cursor per run and k output identities.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `run_partition` (exhaustive): Every valid candidate appears in exactly one run with its original score and identity.
- `head_invariant` (exhaustive): A run head dominates every remaining candidate in that run under the same total order.
- `winner_progress` (exhaustive): Each round emits the best remaining valid candidate and advances exactly its cursor without overflowing cursor storage.
- `collective_realisation` (test): The collective sees all offered heads and resolves equal scores by the same identity order without inactive-lane contamination.

Selection compares the supplied scores exactly; upstream rounding or nonlinear transformation can change ties and needs its own contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
