# cyclic_tile_reduction_partition@1: cyclic tile reduction partition

## When to consider

When split-K should interleave consecutive K tiles across independent partial owners, select participant count and tile width separately. Advance each owner by tile_width*participants, retain its scratch identity and combine after all owners finish. Packed operand byte strides follow their own tile sizes; nonlinear epilogues belong after the complete sum.

## Computation before and after

Parallel reduction participants own cyclic tiles rather than contiguous reduction ranges.

Participant s owns logical tiles s+j*participants and within each owns k=(s+j*participants)*tile_size+u for u in [0,tile_size), masked by k<length. Accumulate those products in increasing j, store a separate partial indexed by s, then combine every participant exactly once. The matmul instance restores packed operand offsets independently: W tile increment is packed_tile_K*participants while logical X increment is tile_size*participants. Apply bias only to s=0 or after final combine; defer nonlinear activation and output quantization to the final reduction pass. Empty participants publish neutral partials instead of reading invalid data.

Recover the original computation: For each logical k recover tile=k//tile_size, owner=tile%participants, iteration=tile//participants and within-tile=k%tile_size; combine all owned products once.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| participants | compile / software | {"lo": 1, "hi": 65536} | Independent partial-result participants. |
| tile_size | compile / software | {"lo": 1, "hi": 65536} | Consecutive logical reduction indices per owned tile. |

## Coupled constraints and scope

Tile-cyclic split-K and equivalent reductions with separate participant-indexed scratch. Existing reduction_partition@1 and @2 name strided threads/warps and contiguous ranges, but do not expose the cyclic tile width/participant map as a compiled assignment.

Scope predicate: `(p.associative_commutative_exact_combine and p.changed_float_order_admitted) and (p.tail_mask_identity and p.empty_participants_emit_identity and p.partial_generation_complete) and (p.additive_epilogue_once_and_nonlinear_epilogue_after_combine)`.

- Precondition `pre0` (semantic): `p.associative_commutative_exact_combine and p.changed_float_order_admitted` — The requested combine admits this exact-arithmetic partition and its floating-point order.
- Precondition `pre1` (semantic): `p.tail_mask_identity and p.empty_participants_emit_identity and p.partial_generation_complete` — Tails/empty owners contribute the identity and final combine waits for all partials.
- Precondition `pre2` (semantic): `p.additive_epilogue_once_and_nonlinear_epilogue_after_combine` — Bias/additive checkpoints occur once; nonlinear activation/quantization waits for the combined value.

## Local example and misuse

For a local instance of `reduce`, identify the object `cyclic_tile_reduction_partition` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/triton/delta/cyclic_tile_reduction_partition@1/template.yaml`. Original outline: `knowledge/studies/triton/delta/cyclic_tile_reduction_partition@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `cyclic_cover` (test): Every logical reduction index has exactly one owner and one iteration, including the final partial tile.
- `partial_and_epilogue` (test): All partials complete before combination; additive and nonlinear epilogues have the required exact-once placement.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
