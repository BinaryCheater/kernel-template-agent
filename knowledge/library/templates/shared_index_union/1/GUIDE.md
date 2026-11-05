# shared_index_union@1: shared union descriptor

## When to consider

Use when two consumers request sorted unique indices with the same payload meaning; choose items_per_split and merge each equal key into one shared entry. Keep each row's original slot as a split base plus offset so results scatter back independently, for example lists[0,2,4] and[1,2,4] share keys2/4 while retaining slots1/2 in both rows. Require compatible physical mappings before composing a paged gather and ignore every padded absent slot.

## Computation before and after

Sorted two-row union with split-local incidence maps and exact scatter recovery.

Merge two sorted index lists, emit each union key once, and retain each consumer's original slot or absence. Split the union into chunks of items_per_split. At the first union key of each split record each row's current insertion cursor as slot_base. For each present row encode original_slot-slot_base; encode absence as65535. Consumer scatter writes its result to slot_base+slot_offset only when present. Pad unused final-split entries with both rows absent. Logical-to-physical translation and actual payload gathering are separate refinements.

Recover the original computation: Scatter the shared indexed results through each row's incidence map to recover that consumer's original index order.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| items_per_split | compile / software | {"lo": 1, "hi": 8192} | Maximum union indices represented by one descriptor split; a partial final split is padded. |

## Coupled constraints and scope

Two sorted unique logical-index lists with at most4096 entries each and reusable equal-key payloads. The union can share data loads or input preparation; consumer-specific arithmetic remains separate. Paged users must have compatible page tables for paired rows; unaligned sparse blocks must merge by resolved token starts rather than by incomparable block IDs. This rule does not claim a speedup; overlap, merge cost and payload reuse require measurements.

Scope predicate: `p.num_consumers == 2 and p.sorted_unique_rows and p.same_index_meaning`.

- Precondition `pre_sorted` (semantic): `p.sorted_unique_rows and p.indices_valid` — Each input row is a strictly increasing list of valid logical source indices; either may be empty.
- Precondition `pre_compatible` (semantic): `p.same_index_meaning and p.payload_reusable` — Equal keys denote the same source payload for both consumers.
- Constraint `c_offsets` (implementation): `p.max_row_items <= 4096` — Source-bounded row slots fit the intermediate15-bit slot and final16-bit offset representation.
- Constraint `c_pair` (implementation): `p.num_consumers == 2` — This version stores two independent incidence maps.

## Local example and misuse

For a local instance of `work_assign`, identify the object `shared_union_descriptor` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/deepgemm/delta/shared_index_union@1/template.yaml`. Original outline: `knowledge/studies/deepgemm/delta/shared_index_union@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Union keys plus two slot offsets per key and two bases per split.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `union_once` (exhaustive): Every key in either row appears exactly once in the shared union and no other key appears.
- `row_slot_identity` (exhaustive): Each original row slot is recovered exactly once from its own base and offset; absence never creates an output.
- `split_padding` (exhaustive): Tail padding carries absent slots for both consumers and cannot write a logical result.
- `publication` (test): Descriptor payload and slot maps are visible before their consumers read them and remain live throughout use.

No arithmetic change for independent consumers; any reordered reduction or changed consumer precision is a separate step.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- enables: indexed_gather (outside menu); Each union key supplies one shared gather coordinate; consumer incidence maps retain their distinct output slots.

Structured contract: [template.yaml](template.yaml).
