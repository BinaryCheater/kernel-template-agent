# spare_operand_slots_epilogue_alias@1: spare operand slots epilogue alias

## When to consider

When an operand ring has two spare slots after next-tile prefetch, borrow them for output staging instead of a separate C allocation. For S=4, three next B tiles occupy a five-slot ring and the next two slots are free; a 128x256 output can split into two 128x128 slices fitting 64x256 B slots. Reject a different prefetch count or refill before output stores finish reading.

## Computation before and after

Output staging borrows one or two free slots in an operand ring after the previous MMA retires.

After draining old MMA reads and issuing S-1 next-tile input generations, let producer be the next input sequence. B ring has S+1 slots; live next-tile operands occupy producer-(S-1) through producer-1 modulo S+1. Borrow producer modulo S+1 and optionally producer+1 modulo S+1 for finalized output. One-slice mode uses BM=BK; two-slice mode uses BM=2*BK and splits output along N into two byte-equivalent B-slot shapes. Fence shared output writes and issue asynchronous stores. The next compute may consume a ready old-prefetched input, but must wait output source reads before the next destructive TMA load. Producer and consumer sequence counters continue across tile boundaries.

Recover the original computation: Give output its own temporary buffer and retain the same input generations; disjoint live slots and completion order preserve all input/output values.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| S | compile / software | {"lo": 1, "hi": 65536} | Input prefetch ring slots; source uses at least three. |
| output_slices | compile / software | ["one", "two"] | One B-sized output tile or two N-half slices sharing the spare slots. |

## Coupled constraints and scope

The source cross-tile Gluon matmul pipeline with B ring S+1 and next prologue S-1, retired old MMA reads, output BM=BK or BM=2*BK. Not an arbitrary ring alias independent of schedule.

Scope predicate: `(S >= 3 and p.B_ring_slots == S + 1 and p.next_tile_live_B_slots == S - 1) and (p.previous_mma_reads_retired and p.next_tile_prefetch_occupies_preceding_sequence_slots) and (((output_slices == 'one' and p.BM == p.BK) or (output_slices == 'two' and p.BM == 2 * p.BK and p.BN % 2 == 0)) and p.output_slice_bytes_fit_B_slot and p.output_slice_layout_legal and p.next_load_waits_output_source_reads)`.

- Precondition `pre0` (semantic): `S >= 3 and p.B_ring_slots == S + 1 and p.next_tile_live_B_slots == S - 1` — B ring has exactly two spare slots after the next prologue.
- Precondition `pre1` (semantic): `p.previous_mma_reads_retired and p.next_tile_prefetch_occupies_preceding_sequence_slots` — Old operand reads are complete and next inputs occupy the preceding S-1 slots.
- Precondition `pre2` (semantic): `((output_slices == 'one' and p.BM == p.BK) or (output_slices == 'two' and p.BM == 2 * p.BK and p.BN % 2 == 0)) and p.output_slice_bytes_fit_B_slot and p.output_slice_layout_legal and p.next_load_waits_output_source_reads` — Each slice fits its borrowed slot and no following load overwrites a pending store source.

## Local example and misuse

For a local instance of `store`, identify the object `spare_operand_slots_epilogue_alias` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/triton/delta/spare_operand_slots_epilogue_alias@1/template.yaml`. Original outline: `knowledge/studies/triton/delta/spare_operand_slots_epilogue_alias@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `live_slot_disjointness` (proof): The two future sequence slots do not overlap the S-1 live preceding B slots modulo S+1.
- `output_coordinates` (test): Split slices preserve each original output coordinate and fit borrowed allocation/layout.
- `retirement` (test): Previous MMA and subsequent TMA/store events obey alias lifetime.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
