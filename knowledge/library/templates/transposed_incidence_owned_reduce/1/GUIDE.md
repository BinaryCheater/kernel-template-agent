# transposed_incidence_owned_reduce@1: transposed incidence owned reduce

## When to consider

When indexed scatter contributions would contend, materialize inverse incidence and let each destination own its complete gather-reduction. One input participates in two max outputs that both equal it: gather both upstream gradients into that input. A tied second input must also receive its own edge gradient under tie-all semantics. Dropping one tied edge changes the contract.

## Computation before and after

When indexed scatter contributions would contend, materialize inverse incidence and let each destination own its complete gather-reduction.

Replace contributions sent along forward incidence with per-destination traversal of inverse incidence slots. For each destination feature, initialize its accumulator, loop all offset_count slots, load only present source rows, apply the original edge predicate/weight, reduce locally and store once. For max-pool backward the predicate is input_value==forward_output, so all tied maxima receive their original upstream contributions rather than selecting one winner. For average backward a correct gradient requires upstream_gradient/present_count; a different source multiplier cannot instantiate that gradient contract.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| offset_count | compile / software | {"lo": 1, "hi": 4096} | Number of stencil relation slots visited per destination; workload-fixed for a binding. |

## Coupled constraints and scope

Sparse NVIDIA stencil reductions where inverse incidence can be built once and reused across channels. The construction changes ownership, not the operator or its gradient convention; max tie-all and argmax-only gradients are different contracts. Resource effects: Inverse map O(offset_count*destinations); one output store per feature replaces contended writes, at the cost of reading neighbor slots including absent entries.

Scope predicate: `p.inverse_incidence_exact and p.invalid_sentinel_guarded and p.destination_owner_unique and p.inputs_immutable and p.numeric_order_admitted and p.edge_weight_contract_explicit`.

- Precondition `pre_0` (semantic): `p.inverse_incidence_exact and p.invalid_sentinel_guarded` — Inverse incidence preserves every logical edge once and excludes -1 before payload access.
- Precondition `pre_1` (semantic): `p.destination_owner_unique and p.inputs_immutable` — One executor owns each destination feature; source payloads do not change during reduction.
- Precondition `pre_2` (semantic): `p.numeric_order_admitted and p.edge_weight_contract_explicit` — Caller admits chosen serial reduction order and exact per-edge weighting/predicate.

## Local example and misuse

For a local instance of `map`, identify the object `transposed_incidence_owned_reduce` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/spconv/delta/transposed_incidence_owned_reduce@1/template.yaml`. Original outline: `knowledge/studies/spconv/delta/transposed_incidence_owned_reduce@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Inverse map O(offset_count*destinations); one output store per feature replaces contended writes, at the cost of reading neighbor slots including absent entries.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `transpose` (proof): Each forward edge corresponds to exactly one inverse edge and neither duplicates nor disappears.
- `edge_values` (proof): Each gathered contribution uses original source feature, predicate and weight.
- `exclusive_write` (proof): No other executor writes the same destination feature, so the final store needs no atomic.
- `boundaries` (proof): Absent neighbors contribute the reduction identity; empty neighborhoods follow the declared initial-value contract.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
