# cyclic_warp_row_skew@1: cyclic warp row skew

## When to consider

When a cross-warp reduction reads same-bank rows from several warp partials, shift each warp row digit cyclically before storing. Choose skew zero for identity or one for the observed outer-statistic layout; preserve the inverse reader map. This exposes addition modulo row-count rather than mislabeling it an XOR swizzle.

## Computation before and after

A warp-indexed shared tile cyclically shifts each warp row digit to vary its bank placement.

Let R=32/row_width. A logical element indexed by chunk,warp,row,column uses address chunk*p.block_threads+warp*32+((row+warp_skew*warp)%R)*row_width+column, with 0<=row<R and 0<=column<row_width. Invert row=(stored_row-warp_skew*warp)%R. Choose warp_skew to change peer-warp bank incidence without changing values; validate the actual reader request footprint before claiming fewer bank passes.

Recover the original computation: Invert the cyclic shift for each warp and recover the original logical shared tile.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| row_width | compile / software | {"lo": 1, "hi": 65536} | Contiguous elements per row digit; divides the NVIDIA 32-lane warp. |
| warp_skew | compile / software | {"lo": 0, "hi": 31} | Coefficient of warp index in cyclic row shift; source binds one. |

## Coupled constraints and scope

NVIDIA warp-indexed shared tiles with row_width dividing32 and matching read/write mapping. Source row_width8 gives R4,warp_skew1. No universal conflict-free guarantee is encoded.

Scope predicate: `32 % row_width == 0 and warp_skew < 32 / row_width and p.writer_reader_use_same_skew and p.shared_lifetime_valid`.

- Precondition `pre0` (semantic): `p.writer_reader_use_same_skew and p.shared_lifetime_valid` — Writes and reads agree on the same address permutation and lifetime.
- Constraint `c0` (semantic): `32 % row_width == 0 and warp_skew < 32 / row_width` — Within-warp row digits fit the lane decomposition.

## Local example and misuse

For a local instance of `access`, identify the object `cyclic_warp_row_skew` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/fuser/delta/cyclic_warp_row_skew@1/template.yaml`. Original outline: `knowledge/studies/fuser/delta/cyclic_warp_row_skew@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Source-bound physical buffer; exact byte count and capacity supplied by binding.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `address_bijection` (proof): Cyclic row addition is a permutation independently within each warp chunk.
- `bank_footprint` (test): Chosen skew achieves its claimed bank behavior for the actual vector/read grouping.

Address/storage transformation preserves values; arithmetic changes require separate obligations.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
