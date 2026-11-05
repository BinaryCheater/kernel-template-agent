# pointwise_slice_demand_pushdown@1: pointwise slice demand pushdown

## When to consider

When a pointwise producer computes values later discarded by a slice, propagate the demanded coordinates backward instead of materializing the full producer output. For y[j]=F(x[j+L]), compute only that input interval. Clone a shared producer for the sliced consumer so an unsliced consumer still gets its full domain; do not move a padding constant through F.

## Computation before and after

A slice after a pure pointwise producer narrows producer evaluation to the requested input-coordinate domain.

Replace slice(F(x),start,length) with evaluating the same pointwise F only at original coordinates start+j for j in [0,length). Propagate that demanded coordinate interval backward through each pure pointwise producer without changing its scalar operations. If another consumer needs the unsliced producer domain, keep its original producer and clone the producer slice for this consumer. Padding is not pushed through F by this version; preserve the original padding value with a separate guarded output path.

Recover the original computation: Substitute the original coordinate start+j into the unchanged scalar producer DAG to recover every sliced result, leaving other consumers unchanged.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| start | runtime / boundary | {"lo": 0, "hi": 2147483647} | First retained logical input coordinate; supplied by the slice contract. |

## Coupled constraints and scope

Pure coordinate-local pointwise producer DAG followed by an in-bounds contiguous slice. No reduction, RNG, mutation or arbitrary pad-through-nonlinearity transformation. Native resize scheduler has additional shape restrictions; this candidate records the observed slice mechanism.

Scope predicate: `start + p.output_length <= p.input_length and p.output_length >= 0 and p.producer_pure_pointwise_and_coordinate_local and p.input_mapping_preserved and p.other_consumers_keep_original_producer_domain and p.replay_numeric_semantics_preserved`.

- Precondition `pre0` (semantic): `p.producer_pure_pointwise_and_coordinate_local and p.input_mapping_preserved` — Each requested output depends only on corresponding producer input coordinates.
- Precondition `pre1` (semantic): `p.other_consumers_keep_original_producer_domain and p.replay_numeric_semantics_preserved` — Shared producers are cloned as needed and arithmetic/casts retain their contract.
- Constraint `c0` (semantic): `start + p.output_length <= p.input_length and p.output_length >= 0` — Demand interval is within the producer output domain.

## Local example and misuse

For a local instance of `map`, identify the object `pointwise_slice_demand_pushdown` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/fuser/delta/pointwise_slice_demand_pushdown@1/template.yaml`. Original outline: `knowledge/studies/fuser/delta/pointwise_slice_demand_pushdown@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Source-bound physical buffer; exact byte count and capacity supplied by binding.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `coordinate_identity` (proof): Each retained output evaluates the same original-coordinate producer inputs.
- `consumer_isolation` (test): Cloning/narrowing does not change another consumer required domain or values.

Address/storage transformation preserves values; arithmetic changes require separate obligations.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
