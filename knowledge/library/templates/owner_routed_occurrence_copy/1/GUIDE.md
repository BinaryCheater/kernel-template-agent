# owner_routed_occurrence_copy@1: owner routed occurrence copy

## When to consider

When routed expert work needs to move token/expert occurrences across shards, retain a unique occurrence row and its expert owner. Forward fanout copies a token to every selected expert row; the inverse route copies each expert result to its own destination occurrence. Publish peer stores before consumers and reduce occurrences only in an explicit later stage.

## Computation before and after

Selected logical occurrences are copied to uniquely owned peer rows and published as a completed generation.

For token_to_expert, each source rank enumerates its local token t, restores global token g=t+rank*local_tokens, and loads selected expert e(g,j) plus destination row gate(g,j). Compute owner(e) from the unique membership bit across ranks; the source uses sum(rank*membership). Copy the token vector to peer[owner(e)][gate(g,j)] for every selected occurrence j, retaining duplicates only if their destination rows are distinct. For expert_to_occurrence, source row s carries a global destination occurrence d and its expert e(d). Only rank owner(e(d)) writes s to peer[d/local_occurrences][d%local_occurrences]. Launch completion plus the symmetric-memory barrier publishes the whole destination generation. A later reduction across expert occurrences is separate; ordinary stores here must not be presented as additive scatter.

Recover the original computation: Erase placement by following each output occurrence back to its original token/expert occurrence; fanout preserves one value per selected occurrence, inverse copy preserves its own expert result.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| direction | compile / software | ["token_to_expert", "expert_to_occurrence"] | Forward fanout or inverse occurrence permutation; neither direction adds expert outputs. |

## Coupled constraints and scope

Direct peer-addressed copies with explicit occurrence identity and unique expert owners. Replicated expert ownership, duplicate destination rows or reduction-to-token require another protocol. The inverse global row is an occurrence index in the flattened selected-expert domain, not necessarily a token index.

Scope predicate: `(p.expert_owner_partition_exact and p.peer_buffers_addressable) and (p.destination_occurrence_map_injective and p.all_indices_valid and p.rows_cover_requested_occurrences) and (p.old_readers_retired and p.peer_store_completion_barrier_before_consumption)`.

- Precondition `pre0` (semantic): `p.expert_owner_partition_exact and p.peer_buffers_addressable` — Each expert has exactly one owner rank; rank0 is a real owner, not missing metadata.
- Precondition `pre1` (semantic): `p.destination_occurrence_map_injective and p.all_indices_valid and p.rows_cover_requested_occurrences` — Every peer destination row has one writer and all requested occurrences are represented.
- Precondition `pre2` (semantic): `p.old_readers_retired and p.peer_store_completion_barrier_before_consumption` — Buffers are fresh/reusable and the communication runtime publishes all peer writes before consumers read.

## Local example and misuse

For a local instance of `transfer`, identify the object `owner_routed_occurrence_copy` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/triton/delta/owner_routed_occurrence_copy@1/template.yaml`. Original outline: `knowledge/studies/triton/delta/owner_routed_occurrence_copy@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `owner_and_address` (test): Unique expert ownership and injective occurrence indices yield exactly one writer per peer destination coordinate.
- `generation_publication` (test): Consumers see complete stores only after runtime barrier completion and no prior generation is reused early.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
