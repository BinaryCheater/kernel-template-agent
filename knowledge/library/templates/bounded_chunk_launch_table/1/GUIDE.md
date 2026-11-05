# bounded_chunk_launch_table@1: bounded chunk launch table

## When to consider

When many small contiguous tensors share an elementwise operation, batch tensor/chunk descriptors into bounded kernel arguments to amortize launches without padding tensors to one size.

## Computation before and after

When many small contiguous tensors share an elementwise operation, batch tensor/chunk descriptors into bounded kernel arguments to amortize launches without padding tensors to one size.

For each nonempty tensor t enumerate chunks c=0..ceil(n_t/C)-1. Append descriptor (local_tensor_slot,c). Flush when the tensor metadata or block descriptor capacity is reached, or after the final chunk. If a flush splits a tensor, copy its metadata into slot0 of the next batch and continue with the original chunk number. A block reads descriptor b, uses each operand base plus c*C, and guards local i by both i<C and c*C+i<n_t. Metadata is passed by value so later host batches do not overwrite live arguments. Preserve per-tensor scalar metadata when rebasing slots.

Recover the original computation: Host-built descriptor batches preserve the concatenation of independent per-tensor element operations; no cross-tensor reduction or alias-dependent update order is implied.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| chunk_size | compile / software | {"lo": 1, "hi": 2147483647} | C: free positive chunk extent. |
| tensor_capacity | compile / software | {"lo": 1, "hi": 255} | Metadata capacity selected for operand depth and kernel argument budget. |
| block_capacity | compile / software | {"lo": 1, "hi": 320} | Descriptor capacity selected for argument budget. |

## Coupled constraints and scope

Host-built descriptor batches preserve the concatenation of independent per-tensor element operations; no cross-tensor reduction or alias-dependent update order is implied.

Scope predicate: `(p.metadata_bytes <= hw.kernel_argument_bytes) and (p.nonempty_tensors and p.contiguous_matching_operand_sizes and p.metadata_stable and p.independent_chunk_writes)`.

- Precondition `pre0` (semantic): `p.nonempty_tensors and p.contiguous_matching_operand_sizes and p.metadata_stable and p.independent_chunk_writes` — Positive tensor lengths, same-device contiguous operands, stable descriptors, independent output ownership.
- Constraint `c0` (implementation): `p.metadata_bytes <= hw.kernel_argument_bytes` — All arguments including functor/scalars fit target launch limit.

## Local example and misuse

For a local instance of `work_assign`, identify the object `bounded_chunk_launch_table` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/apex/delta/bounded_chunk_launch_table@1/template.yaml`. Original outline: `knowledge/studies/apex/delta/bounded_chunk_launch_table@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `coverage` (proof): Every tensor/chunk pair appears once across flushes; carried metadata does not reset chunk index.
- `addressing` (proof): Local slot rebasing preserves original tensor and every guarded element address.

Floating-point reassociation, casts and approximate functions require an admitted numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
