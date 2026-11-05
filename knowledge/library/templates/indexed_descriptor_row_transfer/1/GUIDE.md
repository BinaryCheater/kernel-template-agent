# indexed_descriptor_row_transfer@1: indexed descriptor row transfer

## When to consider

When rows are irregular but each row needs the same contiguous column window, use an indexed descriptor transfer instead of scalar gather loads. Choose direction, row count and column width subject to hardware layout/alignment. For FP16, cols must be at least16 and column origin a multiple of8 elements. Preserve repeated gather occurrences; reject overlapping scatter destinations without a separate contract.

## Computation before and after

One descriptor transfer gathers or scatters a list of independently indexed rows with one shared contiguous column interval.

Form row index vector offsets[r] and a common column origin y. Gather maps shared[r,c] to global[offsets[r],y+c], zero filling out-of-range coordinates. Scatter maps dense shared[r,c] to that global address, skipping permitted positive out-of-range coordinates; negative scatter coordinates are illegal. Convert row offsets to the source-required replicated lane layout, allocate matching shared layout and for gather register rows*descriptor_block_bytes on the mbarrier before waiting for completion. Scatter fences generic shared writes before TMA and waits source-read completion before reuse. Duplicated row offsets are allowed for gather but require an explicit separate conflict policy for scatter.

Recover the original computation: Resolve every row index and column offset to the original logical global coordinate; dense result rows retain their occurrence identity.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| direction | compile / software | ["gather", "scatter"] | Read indexed global rows into a dense shared tile or write a dense shared tile to indexed global rows. |
| rows | compile / software | {"lo": 1, "hi": 65536} | Number of row offsets, at least eight in the source. |
| cols | compile / software | {"lo": 1, "hi": 65536} | Transferred contiguous columns per row. |

## Coupled constraints and scope

Blackwell NVIDIA TMA gather/scatter, two-dimensional descriptor with block [1,cols], rows>=8, cols >= (32/element_bits)*8 and 16-byte aligned column origin. Gather accepts negative offsets as zero-fill; scatter does not. Not an arbitrary paged tensor transfer.

Scope predicate: `(rows >= 8 and p.descriptor_rank_two_and_block_rows_one and p.row_offset_layout_valid) and (p.column_offset_bytes % 16 == 0 and cols >= (32 / p.element_bits) * 8) and (direction == 'gather' or p.scatter_offsets_nonnegative_and_destinations_disjoint) and (p.descriptor_and_shared_layout_match and p.transfer_completion_before_read_or_reuse)`.

- Precondition `pre0` (semantic): `rows >= 8 and p.descriptor_rank_two_and_block_rows_one and p.row_offset_layout_valid` — Blackwell source descriptor is [1,cols], with replicated lane bases and required register bases for offsets.
- Precondition `pre1` (semantic): `p.column_offset_bytes % 16 == 0 and cols >= (32 / p.element_bits) * 8` — Column origin is 16-byte aligned and source dtype-dependent extent is legal.
- Precondition `pre2` (semantic): `direction == 'gather' or p.scatter_offsets_nonnegative_and_destinations_disjoint` — Scatter excludes negative offsets and requires conflict-free destination rows.
- Precondition `pre3` (semantic): `p.descriptor_and_shared_layout_match and p.transfer_completion_before_read_or_reuse` — Descriptor layout and asynchronous completion protect the dense shared tile.

## Local example and misuse

For a local instance of `transfer`, identify the object `indexed_descriptor_row_transfer` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/triton/delta/indexed_descriptor_row_transfer@1/template.yaml`. Original outline: `knowledge/studies/triton/delta/indexed_descriptor_row_transfer@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `index_identity` (proof): Each dense row occurrence uses exactly its requested global row and common column interval.
- `layout_and_bounds` (test): Native offset register/lane layout, alignment and asymmetric gather/scatter bounds are obeyed.
- `completion` (test): Payload visibility and source/destination lifetimes match the asynchronous direction.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
