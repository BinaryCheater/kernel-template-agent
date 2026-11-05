# ordered_transmittance_batch_summary@1: ordered transmittance batch summary

## When to consider

When front-to-back compositing has long candidate lists, compute independent consecutive batch summaries and compose them in depth order, replaying the first threshold-crossing batch for exact stopping. World3DGS parallel source uses partials, batch-scan, then replay; forward-only fallback has a different stopping approximation.

## Computation before and after

When front-to-back compositing has long candidate lists, compute independent consecutive batch summaries and compose them in depth order, replaying the first threshold-crossing batch for exact stopping.

For a batch from fresh T=1 compute local C=sum_i alpha_i*product_before_i*c_i and Q=product_i(1-alpha_i). Fold earlier (C0,Q0) then later (C1,Q1) as (C0+Q0*C1,Q0*Q1). Never permute batches. If cumulative Q times next batch Q exceeds threshold, fold directly. Otherwise preserve batch-start C,Q and replay that batch candidate-by-candidate with the original threshold include/exclude policy; discard later batches. Store cumulative batch checkpoints for independent reverse processing and atomic gradient combination. A full-batch fallback is a separate approximate contract.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| batch_size | compile / software | {"lo": 1, "hi": 65533} | Free consecutive candidate count per batch; source logical batch is tile area. |

## Coupled constraints and scope

Real affine compositing or explicitly admitted floating regrouping; nonnegative alpha below one for inverse replay. Exact stopping requires replay, not full crossing-batch fold. Resource effects: Parallel CTA per batch and O(batches*pixels*(channels+1)) checkpoint storage.

Scope predicate: `p.ordered_candidates and p.alpha_in_unit_interval and p.same_acceptance_predicates and p.rounding_admitted and p.threshold_policy_bound and p.summary_publication_ordered`.

- Precondition `pre_0` (semantic): `p.ordered_candidates and p.alpha_in_unit_interval and p.same_acceptance_predicates` — Candidate order and per-candidate alpha are independent of accumulated color.
- Precondition `pre_1` (semantic): `p.rounding_admitted and p.threshold_policy_bound and p.summary_publication_ordered` — Arithmetic regrouping and exact include/exclude saturation policy are explicit.

## Local example and misuse

For a local instance of `map`, identify the object `ordered_transmittance_batch_summary` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/gsplat/delta/ordered_transmittance_batch_summary@1/template.yaml`. Original outline: `knowledge/studies/gsplat/delta/ordered_transmittance_batch_summary@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Parallel CTA per batch and O(batches*pixels*(channels+1)) checkpoint storage.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `summary` (proof): Each batch summary denotes the same affine color/throughput action on any incoming C,T.
- `crossing` (proof): First crossing batch replay recovers exact accepted prefix; threshold equality follows policy.
- `checkpoint` (proof): All checkpoints are published before backward; gradients of shared inputs are combined without loss.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
