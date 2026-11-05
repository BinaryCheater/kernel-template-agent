# descriptor_union_completion_count@1: descriptor union completion count

## When to consider

When multicast MMA has several operand replication patterns, derive completion arrivals from the union of elected CTA issuers. A CTA elected by both A and B still sends one commit arrival; two-CTA mode counts only its even-rank atom owner. Registering per-descriptor totals can deadlock, while counting too few allows premature reuse.

## Computation before and after

Expected MMA completion arrivals are derived from the union of descriptor-driven issuing CTA sets.

Enumerate candidate CTA ranks c in the cluster; in two_ctas mode skip odd c. A descriptor admits c when every nonzero CGA basis bit position i has (c & (1<<i))==0; zero bases impose no restriction. Count c once if any descriptor admits it. With no descriptors or multicast disabled use one arrival. Initialize stage-release and accumulator-ready barriers using this derived unique issuer count; do not multiply arrivals by the number of descriptors or assume count equals physical CTA count.

Recover the original computation: Replace the barrier arrival count by the number of actual unique issuing CTA atoms whose engine completion the receiver joins.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| instruction_group | compile / software | ["one_cta", "two_ctas"] | Two-CTA atoms elect only even CTA ranks in the source. |

## Coupled constraints and scope

The Gluon tcgen05 multicast helper with up to four shared descriptors and matching cluster bit count. Derived arrival count is not an optimizer knob; cluster layout and instruction group are inputs.

Scope predicate: `(p.descriptors_share_cluster_bit_count and p.cga_bases_match_payload_identity) and (p.backend_commit_issuers_match_source_union and p.barrier_count_uses_derived_union_size)`.

- Precondition `pre0` (semantic): `p.descriptors_share_cluster_bit_count and p.cga_bases_match_payload_identity` — Descriptor CGA bases describe the same physical cluster and actual operand replication.
- Precondition `pre1` (semantic): `p.backend_commit_issuers_match_source_union and p.barrier_count_uses_derived_union_size` — Barrier registration counts unique issuers, not per-operand multiplicity.

## Local example and misuse

For a local instance of `sync`, identify the object `descriptor_union_completion_count` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/triton/delta/descriptor_union_completion_count@1/template.yaml`. Original outline: `knowledge/studies/triton/delta/descriptor_union_completion_count@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `union_count` (proof): Each actual source issuer is counted once even when several descriptors elect the same CTA.
- `backend_incidence` (test): Descriptor predicate matches actual native issuer/recipient completion routing.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
