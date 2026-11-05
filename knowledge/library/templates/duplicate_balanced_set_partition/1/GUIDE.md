# duplicate_balanced_set_partition@1: duplicate balanced set partition

## When to consider

When parallel set operations encounter duplicate runs crossing tile diagonals, rebalance the cuts through equal-key prefixes before serial intersection/difference/union. Choose input diagonal spacing, not independent A/B cuts. A run with three A copies and two B copies contributes two intersection outputs and one A-minus-B output even when a partition crosses its middle.

## Computation before and after

A merge-diagonal partition adjusted inside equal-key runs so multiset pairs stay consistently assigned.

At diagonal d find ordinary MergePath (i,j). If j<B_length, let x=B[j] and find lower bounds a0,b0 of its equal run in the visited A/B prefixes. Let total=(i-a0)+(j-b0), advanceB=max(total//2,total-(i-a0)), bounded by the available B equal-run end found by upper_bound in the source search range. Set advanceA=total-advanceB. If advanceA=advanceB+1 and another equal B remains, set star=1; otherwise star=0. Return A_boundary=a0+advanceA, B_boundary=d-A_boundary+star. Neighboring corrected boundaries define disjoint source subranges. Apply the requested serial multiset operation locally and compact surviving outputs through exact output-count prefixes. Intersection/difference output payloads come from A; union/symmetric difference preserve their declared source multiplicities.

Recover the original computation: Concatenate local outputs to recover the requested sorted multiset operation, with equal-key counts paired consistently across partition boundaries.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| tile_inputs | compile / software | {"lo": 1, "hi": 65536} | Nominal total input diagonal spacing per parallel tile. |

## Coupled constraints and scope

Two sorted sequences and the explicit balanced-path correction used by CCCL; integer coordinate packing must fit selected backend fields. This differs from a stable merge partition, which emits every duplicate rather than pairing equal occurrences.

Scope predicate: `(p.inputs_sorted_same_strict_weak_order and p.equivalent_key_classes_consistent) and (p.balanced_boundary_algorithm_and_consumers_match and p.output_multiplicity_contract_explicit) and (p.output_prefixes_exact and p.payload_source_matches_operation)`.

- Precondition `pre0` (semantic): `p.inputs_sorted_same_strict_weak_order and p.equivalent_key_classes_consistent` — Both sequences share comparator and equality classes.
- Precondition `pre1` (semantic): `p.balanced_boundary_algorithm_and_consumers_match and p.output_multiplicity_contract_explicit` — All boundaries and local set operations use the same duplicate pairing convention.
- Precondition `pre2` (semantic): `p.output_prefixes_exact and p.payload_source_matches_operation` — Output compaction and key/value source identity match union/intersection/difference.

## Local example and misuse

For a local instance of `map`, identify the object `duplicate_balanced_set_partition` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cccl/delta/duplicate_balanced_set_partition@1/template.yaml`. Original outline: `knowledge/studies/cccl/delta/duplicate_balanced_set_partition@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `duplicate_pairing` (proof): Corrected boundaries do not separate an equal-key matched pair in a way that changes the requested multiplicity.
- `count_and_payload` (test): Output prefixes compact survivors once and payload source follows the operation contract.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
