# cross_buffer_operand_window@1: two chunk operand window

## When to consider

When a fixed consumer instruction window does not divide the producer's storage chunks, consider addressing the window through current and successor chunks. Preserve each logical stream coordinate and keep both chunks live through any crossing read; this version requires W<=C and a legal two-address backend. The pinned SM103 implementation binds W96,C256 with crossing windows2/5; changing C requires matching descriptor, chunk-layout and retirement code rather than a source parameter override.

## Computation before and after

Resolve a contiguous consumer window through a current chunk plus an optional successor.

A consumer window i begins at logical offset i*W and spans W elements. With chunk extent C, derive current=floor(i*W/C) and offset=(i*W)%C. Address the prefix through current at offset and any remainder through successor=current+1 at zero. Supply both addresses to the consumer without concatenating the chunks into a new buffer. Retain readiness and asynchronous completion dependencies; retire a chunk only after the last window touching it has completed.

Recover the original computation: Resolve each window element to its unique logical stream coordinate i*W+k and erase the physical chunk descriptors.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| chunk_extent | compile / software | {"lo": 1, "hi": 2147483647} | Logical elements represented by each producer/storage chunk; independent of the consumer instruction extent. |

## Coupled constraints and scope

Uniform contiguous consumer windows with extent W over equal-sized logical producer chunks C, W<=C. Physical chunk addresses may be noncontiguous, including distinct shared-ring slots. Producer ring rotation, chunk layout, scale metadata and completion events remain separate refinements. No new buffer or extra copy is introduced by this address mapping; target instruction traffic is not priced here.

Scope predicate: `p.window_extent > 0 and p.window_extent <= chunk_extent and p.two_chunk_descriptor_supported and p.chunk_layout_legal and p.chunks_ready_and_live and p.tail_padding_valid`.

- Precondition `pre_two_chunks` (semantic): `p.window_extent > 0 and p.window_extent <= chunk_extent` — The two-address representation covers the whole window.
- Precondition `pre_backend` (implementation): `p.two_chunk_descriptor_supported and p.chunk_layout_legal` — Encoding and target restrictions are established separately for this instance.
- Precondition `pre_lifetime` (semantic): `p.chunks_ready_and_live and p.tail_padding_valid` — Every accessed chunk is ready before consumer use and remains live through completion; logical tail padding follows the original operation.
- Constraint `c_two_chunks` (semantic): `p.window_extent > 0 and p.window_extent <= chunk_extent` — Each consumer window touches at most its current chunk and one successor.
- Constraint `c_backend` (implementation): `p.two_chunk_descriptor_supported and p.chunk_layout_legal` — The target consumer represents both addresses and the chosen chunk shape/alignment legally.

## Local example and misuse

For a local instance of `transfer`, identify the object `two_chunk_operand_window` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/cross_buffer_operand_window@1/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/cross_buffer_operand_window@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Existing producer chunks; no concatenation buffer.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `coordinate_identity` (exhaustive): Each consumer element k maps to exactly logical offset i*W+k through the current or next chunk.
- `last_use` (test): A chunk is not overwritten or released before every consumer window touching it has completed.
- `descriptor_encoding` (unspecified): The selected backend encodes current offset and successor with the required alignment, address spaces and target gates.

Address-only transformation on the stated domain; consumer arithmetic and rounding are unchanged.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- requires: [physical_storage@1](../../physical_storage/1/GUIDE.md); Chunk lifetimes must include every crossing consumer window.

Structured contract: [template.yaml](template.yaml).
