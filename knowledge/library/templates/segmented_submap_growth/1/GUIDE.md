# segmented_submap_growth@1: segmented submap growth

## When to consider

When a hash container grows without rehashing existing entries, add bounded-load submaps and search existing segments before inserting into one designated target segment. Key7 already lives in segment0: inserting it while target1 is active must check0 first. Two concurrent batches targeting different segments can both miss7 then each insert it; local map CAS alone cannot prevent that cross-segment race.

## Computation before and after

When a hash container grows without rehashing existing entries, add bounded-load submaps and search existing segments before inserting into one designated target segment.

Reserve usable capacity by appending fixed-capacity maps, source doubling aggregate capacity while respecting load/headroom policy. For a batch assigned to target t, query all other submaps for each key; if found, report existing or update its payload under requested operation. Otherwise atomically insert into t, using its local duplicate rule for same-phase contenders. Search/retrieve traverses all segments. Old entries remain at their original addresses, so growth adds search work rather than migrating old payloads.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| minimum_insert_batch | compile / software | [10000] | Source insertion headroom threshold; a policy, not a uniqueness condition. |

## Coupled constraints and scope

NVIDIA dynamic_map with stream-ordered host growth and designated insertion submap. Source max_load_factor0.60 and min_insert_size10000 are policy constants. Concurrent independent target phases are outside this version. Resource effects: Retains old payloads without rehash traffic; adds multiple contains probes and a device ref-array upload, with memory reserved in growing segments.

Scope predicate: `p.submap_mutation_phases_ordered and p.target_submap_unique and p.all_submaps_searched and p.local_insert_unique and p.references_live_and_capacity_sufficient`.

- Precondition `pre_0` (semantic): `p.submap_mutation_phases_ordered and p.target_submap_unique` — Each insertion phase has one target submap and no competing phase inserts the same key into another target concurrently.
- Precondition `pre_1` (semantic): `p.all_submaps_searched and p.local_insert_unique` — Every non-target map is checked for an existing key and target insertion enforces local uniqueness.
- Precondition `pre_2` (semantic): `p.references_live_and_capacity_sufficient` — Submap/ref-array storage remains live during device kernels and enough target capacity is reserved.

## Local example and misuse

For a local instance of `map`, identify the object `segmented_submap_growth` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cucollections/delta/segmented_submap_growth@1/template.yaml`. Original outline: `knowledge/studies/cucollections/delta/segmented_submap_growth@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Retains old payloads without rehash traffic; adds multiple contains probes and a device ref-array upload, with memory reserved in growing segments.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `global_uniqueness` (proof): No valid key can be newly inserted into two different submaps under the phase restriction.
- `coverage` (proof): Find/contains/erase/retrieve accounts for every live submap as required by its operation contract.
- `ref_lifetime` (proof): Copied device refs remain valid until every kernel using them completes.
- `capacity` (proof): Load/headroom accounting covers requested insertions; duplicate suppression must not be mistaken for guaranteed free space without allocation.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
