# destination_quotient_transfer@1: destination class payload

## When to consider

When several sparse recipients share a transport locality, choose one payload per locality and expand it after arrival while retaining recipient metadata.

## Computation before and after

When several sparse recipients share a transport locality, choose one payload per locality and expand it after arrival while retaining recipient metadata.

For each source record with logical destinations d, map d to a transport class c(d). Emit one payload per distinct valid class, preserving all logical destination IDs and weights in metadata. At class arrival expand that payload to each original logical recipient. Preserve source ID for the inverse combine; never deduplicate numerical contributions merely because transport was deduplicated.

Recover the original computation: Expand the intermediate records back to the original logical deliveries, retaining their identity and multiplicity.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Sparse fanout whose recipients share identical source payload within a transport locality class; logical recipients are unique per source in this version. Class mapping is fixed by topology. Does not combine distinct expert results.

Scope predicate: `p.payload_identical_within_class and p.expansion_metadata_complete and p.destination_map_immutable`.

- Precondition `pre_contract` (semantic): `p.payload_identical_within_class and p.expansion_metadata_complete and p.destination_map_immutable` — Sparse fanout whose recipients share identical source payload within a transport locality class; logical recipients are unique per source in this version. Class mapping is fixed by topology. Does not combine distinct expert results.

## Local example and misuse

For a local instance of `transfer`, identify the object `destination_class_payload` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/deepep/delta/destination_quotient_transfer@1/template.yaml`. Original outline: `knowledge/studies/deepep/delta/destination_quotient_transfer@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: destination_class_payload. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `class_once` (exhaustive): Exactly one transport payload exists for each source/class pair.
- `expand_exact` (exhaustive): Expansion recovers each original source/destination delivery exactly once, including its weight.
- `publish` (exhaustive): Payload and metadata arrive before any forwarded read or buffer reuse.

No numeric risk is declared by this proposal.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
