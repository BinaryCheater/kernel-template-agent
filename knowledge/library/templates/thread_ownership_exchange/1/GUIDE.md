# thread_ownership_exchange@1: thread ownership exchange

## When to consider

When fused inputs and outputs require different contiguous lane directions, choose an explicit cache boundary to exchange a tile through shared memory. Bind cached_input or cached_output and both thread maps; a 32x32 tile on 256 threads may use four-element vectors on one side and two-element vectors with two unrolled operations on the other. Synchronize before readers switch ownership.

## Computation before and after

A materialized tile at a selected producer/consumer boundary transfers ownership between two thread mappings.

At exchange_boundary allocate shared tile S. Threads of writer mapping W store value V(x) at the shared address for logical coordinate x. Synchronize the participating CTA. Threads with consumer mapping R read S at the address for their required x, then release the tile only after all readers retire. Input and output vector widths may differ provided vector_width*unroll equals each thread mapping assigned element count.

Recover the original computation: Erase shared storage and the ownership maps: consumer of coordinate x reads the same logical V(x) written at the chosen boundary.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| exchange_boundary | compile / software | ["cached_input", "cached_output"] | Observed placement alternatives where the fusion switches thread ownership. |

## Coupled constraints and scope

CTA-local materialized tensor tiles with two explicit bijective ownership maps and legal vector accesses. This is not the metadata-only tensor_axis_permutation@1 and not the GEMM-specific shared_memory_tiling@1.

Scope predicate: `p.writer_map_bijective and p.reader_map_covers_required_tile and p.tile_bytes <= p.shared_capacity_bytes and p.all_writes_visible_before_reads and p.no_overwrite_before_last_reader and p.all_participants_converge and p.boundary_preserves_value_and_coordinate_semantics`.

- Precondition `pre0` (semantic): `p.all_writes_visible_before_reads and p.no_overwrite_before_last_reader and p.all_participants_converge` — CTA publication, convergence and storage lifetime protect exchange.
- Precondition `pre1` (semantic): `p.boundary_preserves_value_and_coordinate_semantics` — The selected cache boundary contains values matching the consumers at that point.
- Constraint `c0` (semantic): `p.writer_map_bijective and p.reader_map_covers_required_tile` — Each logical tile point has one writer and every consumer can find its required points.
- Constraint `c1` (semantic): `p.tile_bytes <= p.shared_capacity_bytes` — Exchange tile fits shared storage.

## Local example and misuse

For a local instance of `access`, identify the object `thread_ownership_exchange` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/fuser/delta/thread_ownership_exchange@1/template.yaml`. Original outline: `knowledge/studies/fuser/delta/thread_ownership_exchange@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Source-bound physical buffer; exact byte count and capacity supplied by binding.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `coordinate_identity` (proof): Writer and reader shared addresses agree for each logical coordinate.
- `lifetime` (test): Publication precedes every read and retirement precedes every destructive reuse.

Address/storage transformation preserves values; arithmetic changes require separate obligations.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
