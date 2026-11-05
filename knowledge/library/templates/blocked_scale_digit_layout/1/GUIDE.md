# blocked_scale_digit_layout@1: blocked scale digit layout

## When to consider

When a scale consumer expects small column groups interleaved with row subtiles, split coordinates into digits rather than copying a plain transpose. Bind row digit sizes and column group; r=33,c=5 at (32,4,4) maps to [0,1,1,1,1]. Reject consumers using an ordinary row-major or differently ordered digit table.

## Computation before and after

A padded two-dimensional scale table split into row/column digits and physically interleaved.

For logical (r,c), let rt=r//(R_outer*R_inner), ro=r%R_outer, ri=(r//R_outer)%R_inner, ct=c//C, ci=c%C. Store in contiguous digit order [rt,ct,ro,ri,ci]. Offset=rt*p.padded_cols*R_outer*R_inner+ct*R_outer*R_inner*C+ro*R_inner*C+ri*C+ci. Padding storage is separate from logical scale ownership and must not emit scales for nonexistent code groups.

Recover the original computation: Invert the radix digits to r=rt*R_outer*R_inner+ri*R_outer+ro and c=ct*C+ci, recovering the original scale table.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| R_outer | compile / software | {"lo": 1, "hi": 65536} | Fast row digit: row modulo R_outer. |
| R_inner | compile / software | {"lo": 1, "hi": 65536} | Other within-tile row digit: row//R_outer modulo R_inner. |
| C | compile / software | {"lo": 1, "hi": 65536} | Contiguous scale columns per column tile. |

## Coupled constraints and scope

Positive row/column digit sizes, padded allocation and exact integer addressing. Fuser NVFP4 binding R_outer=32,R_inner=4,C=4. Does not change scale values or choose quantization groups.

Scope predicate: `p.padded_rows % (R_outer * R_inner) == 0 and p.padded_cols % C == 0 and p.padded_rows >= p.rows and p.padded_cols >= p.cols and p.scale_coordinates_paired_with_codes and p.consumer_uses_same_digit_layout and p.allocation_lifetime_valid and p.integer_offsets_fit`.

- Precondition `pre0` (semantic): `p.scale_coordinates_paired_with_codes and p.consumer_uses_same_digit_layout` — Producer and decoder agree on the logical scale coordinate.
- Precondition `pre1` (semantic): `p.allocation_lifetime_valid and p.integer_offsets_fit` — Storage and offset arithmetic are valid.
- Constraint `c0` (semantic): `p.padded_rows % (R_outer * R_inner) == 0 and p.padded_cols % C == 0` — Allocation contains whole digit tiles.
- Constraint `c1` (semantic): `p.padded_rows >= p.rows and p.padded_cols >= p.cols` — Every logical scale is inside the padded table.

## Local example and misuse

For a local instance of `access`, identify the object `blocked_scale_digit_layout` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/fuser/delta/blocked_scale_digit_layout@1/template.yaml`. Original outline: `knowledge/studies/fuser/delta/blocked_scale_digit_layout@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Source-bound physical buffer; exact byte count and capacity supplied by binding.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `digit_bijection` (proof): Every valid scale has exactly one physical address and inverse.
- `consumer_pairing` (test): Code and decoded scale use the same logical group despite physical layout.

Address/storage transformation preserves values; arithmetic changes require separate obligations.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
