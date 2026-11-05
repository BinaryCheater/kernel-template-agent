# butterfly_bit_ownership@1: butterfly bit ownership

## When to consider

When a butterfly transform exceeds one thread fragment, group butterflies by register-owned logical bits and exchange ownership between stages; the final store can absorb the remaining permutation.

## Computation before and after

When a butterfly transform exceeds one thread fragment, group butterflies by register-owned logical bits and exchange ownership between stages; the final store can absorb the remaining permutation.

Represent each logical index by log2(H) bits, each owned by exactly one thread coordinate bit or register-value bit. Select a stage list of disjoint logical bit sets whose union is all bits. Before a stage, move its selected bits into register ownership through a bijective exchange (shared memory with synchronization or subgroup shuffles); compute the sum/difference butterfly for each selected bit locally. Maintain the index-to-owner map after every exchange. At the end either restore the initial owner map or use its inverse in the global store. A direct final store is legal only when vector contiguity/alignment and valid-tail predicates follow the current mapping. For nonpower2 input N, use the workload-defined H=next_pow2(N) zero-padded transform then crop to N and apply requested scale.

Recover the original computation: Power-of-two butterfly transforms including explicitly requested zero-pad/crop workloads, source H<=32768. Scaling is part of the operation, not implicitly normalized; no universal numerical bound for arbitrary scales or tree orders is asserted.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| transform_extent | compile / software | {"lo": 2, "hi": 32768} | H: workload-defined padded power-of-two transform extent. |
| elements_per_thread | compile / software | {"lo": 2, "hi": 64} | Register fragment size, power of two; source schedule groups log2(EPT) bits. |

## Coupled constraints and scope

Power-of-two butterfly transforms including explicitly requested zero-pad/crop workloads, source H<=32768. Scaling is part of the operation, not implicitly normalized; no universal numerical bound for arbitrary scales or tree orders is asserted.

Scope predicate: `(is_pow2(transform_extent) and is_pow2(elements_per_thread) and elements_per_thread <= transform_extent) and (p.stage_bit_sets_partition_index_bits and p.exchange_maps_bijective and p.exchange_visibility_complete and p.store_map_matches_final_ownership and p.rounding_admitted)`.

- Precondition `pre0` (semantic): `p.stage_bit_sets_partition_index_bits and p.exchange_maps_bijective and p.exchange_visibility_complete and p.store_map_matches_final_ownership and p.rounding_admitted` — Stage coverage, communication, store mapping and numerical contract.
- Constraint `c0` (implementation): `is_pow2(transform_extent) and is_pow2(elements_per_thread) and elements_per_thread <= transform_extent` — Power-of-two butterfly geometry.

## Local example and misuse

For a local instance of `map`, identify the object `butterfly_bit_ownership` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/quack/delta/butterfly_bit_ownership@1/template.yaml`. Original outline: `knowledge/studies/quack/delta/butterfly_bit_ownership@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `butterfly_coverage` (proof): Every logical butterfly dimension is applied once; disjoint dimension operators commute in exact arithmetic.
- `owner_identity` (proof): Every exchange preserves the logical index attached to each value.
- `direct_store` (proof): Final inverse map writes each requested coordinate once with legal vectors and guarded crop.

Floating-point reassociation, casts and approximate functions require an admitted numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
