# alternating_peer_flit@1: alternating peer flit

## When to consider

When a peer exchange fits a small payload plus a ready tag in one transport unit, alternate two receive slots and retain a local phase bit to distinguish successive messages.

## Computation before and after

When a peer exchange fits a small payload plus a ready tag in one transport unit, alternate two receive slots and retain a local phase bit to distinguish successive messages.

Assign one slot pair to each communicating worker. Source16-byte flit contains <=12-byte payload plus a status word. Local status-bit equality chooses slot1, inequality chooses slot2. Send payload with ready=1 and the selected status; poll local selected slot until ready=1, consume payload, then clear ready and invert its status. Fence reset before reusing the communication slot for later elements. Initialize both slots consistently and keep matching peer iteration/worker maps. Inactive tail threads exchange control flits but do not read/write invalid tensor values. This protocol requires native payload/tag visibility and peer progress; volatile syntax alone is not a portable atomicity proof.

Recover the original computation: Matched NVIDIA peer-memory halo exchanges with two phase-tracked flits per worker; source CC>=700 path. Native memory protocol remains an explicit backend premise, not inferred from vector access alone.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| payload_bytes | compile / software | {"lo": 1, "hi": 12} | Source payload bound inside16-byte flit. |

## Coupled constraints and scope

Matched NVIDIA peer-memory halo exchanges with two phase-tracked flits per worker; source CC>=700 path. Native memory protocol remains an explicit backend premise, not inferred from vector access alone.

Scope predicate: `(p.paired_workers_matching_iteration and p.slots_initialized and p.transport_payload_tag_consistent and p.reset_visible_before_reuse and p.peer_progress_guaranteed)`.

- Precondition `pre0` (semantic): `p.paired_workers_matching_iteration and p.slots_initialized and p.transport_payload_tag_consistent and p.reset_visible_before_reuse and p.peer_progress_guaranteed` — Matched exchanges, initialized phase, native publication/reset contract and resident progress.

## Local example and misuse

For a local instance of `transfer`, identify the object `alternating_peer_flit` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/apex/delta/alternating_peer_flit@1/template.yaml`. Original outline: `knowledge/studies/apex/delta/alternating_peer_flit@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `phase_identity` (proof): Equality/difference selection alternates slots and accepts exactly the current message.
- `payload_visibility` (proof): Observed ready tag implies the complete matching payload, at the native transport granularity.
- `reuse` (proof): Sender cannot overwrite data before receiver consumption/reset permits reuse.

Floating-point reassociation, casts and approximate functions require an admitted numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
