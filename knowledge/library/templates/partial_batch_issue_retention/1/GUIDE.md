# partial_batch_issue_retention@1: pending request batch

## When to consider

When aggregating sparse remote requests and a final nonempty issue must ring a doorbell, retain one batch instead of eagerly draining exact multiples.

## Computation before and after

When aggregating sparse remote requests and a final nonempty issue must ring a doorbell, retain one batch instead of eagerly draining exact multiples.

Compact matching requests in stable lane order into a circular queue of capacity 2*batch_width. After adding at most batch_width requests, issue a full batch only when pending > batch_width. At end issue all remaining pending requests as a final batch that explicitly rings the transport doorbell. Keep each original destination address with its request.

Recover the original computation: Expand the intermediate records back to the original logical deliveries, retaining their identity and multiplicity.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| batch_width | compile / software | {"lo": 1, "hi": 32} | Requests per full issue batch; source binds32 lanes. A different backend needs a matching realization. |

## Coupled constraints and scope

Independent gather requests, rounds add at most batch_width requests, issue leaves at most batch_width pending. Empty streams issue nothing. Transport supports aggregation and explicit final issue. No generic claim that ringing equals remote completion.

Scope predicate: `p.requests_independent and p.queue_capacity >= 2 * batch_width and p.transport_requires_final_doorbell`.

- Precondition `pre_contract` (semantic): `p.requests_independent and p.queue_capacity >= 2 * batch_width and p.transport_requires_final_doorbell` — Independent gather requests, rounds add at most batch_width requests, issue leaves at most batch_width pending. Empty streams issue nothing. Transport supports aggregation and explicit final issue. No generic claim that ringing equals remote completion.

## Local example and misuse

For a local instance of `transfer`, identify the object `pending_request_batch` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/deepep/delta/partial_batch_issue_retention@1/template.yaml`. Original outline: `knowledge/studies/deepep/delta/partial_batch_issue_retention@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: pending_request_batch. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `queue` (exhaustive): No insertion overwrites a pending request and every original request is issued once.
- `final` (exhaustive): If any request exists, the final issue is nonempty, at most batch_width wide, and rings a doorbell before flush.
- `completion` (exhaustive): Flush covers every issued request before consumers read destinations.

No numeric risk is declared by this proposal.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
