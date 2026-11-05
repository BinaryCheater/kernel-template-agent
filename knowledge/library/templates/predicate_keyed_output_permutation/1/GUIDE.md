# predicate_keyed_output_permutation@1: predicate keyed output permutation

## When to consider

When neighboring output rows have different sparse support, sort their predicate keys and retain the row permutation so similar support can be scheduled together without changing output identity. Masks [5,1,4] with filter1 sort by [1,1,0]; whichever equal-key order is chosen, the retained IDs must send rows back to original positions. A second pass filter4 supplies remaining offsets, not a second full convolution.

## Computation before and after

When neighboring output rows have different sparse support, sort their predicate keys and retain the row permutation so similar support can be scheduled together without changing output identity.

Initialize row IDs arange(N), sort support masks together with IDs, comparing either whole mask or mask&offset_filter. Schedule contiguous sorted rows and load/store through retained original IDs. A split pass may mask away offsets outside its filter; combine pass contributions exactly once with defined first-pass initialization and later accumulation. For backward weight reductions, persist aggregate forward tile masks with the forward row tile width so consumers can skip absent offset tiles without changing incidence.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| word_bits | compile / software | [32] | Source mask word width; multiword lexicographic sorting has separate ordering binding. |

## Coupled constraints and scope

NVIDIA sparse convolution mask sorting. Sorting is a schedule choice; stable tie order is unnecessary for independent outputs. Source compiled tile menus and actual benefit are separate from this semantic permutation. Resource effects: Adds row-ID permutation and sorting traffic; similar support may reduce wasted masked contraction tiles, but benefit depends on the workload and target. Forward tile masks add ceil(N/tile_width)*mask_words storage.

Scope predicate: `p.permutation_bijective and p.keys_match_rows and p.offset_partition_exact and p.consumer_maps_identity`.

- Precondition `pre_0` (semantic): `p.permutation_bijective and p.keys_match_rows` — Sort preserves a bijection from scheduled rows to original outputs and each mask describes that row.
- Precondition `pre_1` (semantic): `p.offset_partition_exact and p.consumer_maps_identity` — Split masks partition admitted offsets and consumers use argsort to restore row identity.

## Local example and misuse

For a local instance of `map`, identify the object `predicate_keyed_output_permutation` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/spconv/delta/predicate_keyed_output_permutation@1/template.yaml`. Original outline: `knowledge/studies/spconv/delta/predicate_keyed_output_permutation@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Adds row-ID permutation and sorting traffic; similar support may reduce wasted masked contraction tiles, but benefit depends on the workload and target. Forward tile masks add ceil(N/tile_width)*mask_words storage.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `row_identity` (proof): Every scheduled row maps to exactly one original output and all outputs are covered.
- `support` (proof): Pruned bits exclude only absent or other-pass offsets, never present contributions of this pass.
- `combine` (proof): All offset partitions contribute once and bias/activation is applied at the correct complete-result boundary.
- `checkpoint_shape` (proof): A backward tile mask is consumed with compatible forward mask width and argsort mapping.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
