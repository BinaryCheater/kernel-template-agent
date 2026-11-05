# carry_isolated_bit_counts@1: carry isolated bit counts

## When to consider

When summing many packed membership rows, count several columns at once inside spaced fields, then widen before carries could mix columns. Bind each stage fan-in below its field capacity and restore column phase order explicitly. Tile counts can serve both prefix metadata and final atomic totals.

## Computation before and after

Packed boolean columns are summed in progressively wider carry-isolated fields.

Let a1=min(8,N), a2=min(16,N/a1), a3=N/(a1*a2), requiring integral source stages. For each phase s=0..3 extract (word>>s)&0x11111111 and sum groups of a1 words: each4-bit field holds one column count without carry. Split those counts into phases t=0,1 by (packed>>(4*t))&0x0f0f0f0f; sum a2 groups into8-bit fields. Split by byte phase u=0..3 using (packed>>(8*u))&255 and sum remaining a3 groups into ordinary32-bit counters. Restore original column order by flattening [u,t,s], so column index is8*u+4*t+s. Store tile partial counts and optionally atomically add disjoint tile totals into a fresh zero-initialized final array.

Recover the original computation: Expand each packed field count into the sum of its original input bits, one independent reduction per column.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| rows_per_tile | compile / software | {"lo": 1, "hi": 65536} | Supported power-of-two reduction length with source split factors dividing it. |

## Coupled constraints and scope

Unsigned32-bit membership matrices and source legal stage lengths. Integer exactness requires field/headroom bounds; this is vertical per-column population count, not scalar popcount of each word or bit-transpose alone.

Scope predicate: `(p.uint32_boolean_words and p.rows_split_into_exact_stages) and (p.stage_counts_fit_fields and p.final_integer_counts_do_not_overflow)`.

- Precondition `pre0` (semantic): `p.uint32_boolean_words and p.rows_split_into_exact_stages` — Each input word holds32 independent boolean columns and source stage factors partition all rows.
- Precondition `pre1` (semantic): `p.stage_counts_fit_fields and p.final_integer_counts_do_not_overflow` — First counts<=8 fit4 bits, second counts<=128 fit8 bits, final counts fit their integer destination.

## Local example and misuse

For a local instance of `reduce`, identify the object `carry_isolated_bit_counts` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/triton/delta/carry_isolated_bit_counts@1/template.yaml`. Original outline: `knowledge/studies/triton/delta/carry_isolated_bit_counts@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `no_field_carry` (proof): Every packed intermediate count stays strictly below the next field radix and cannot affect an adjacent column.
- `column_identity` (test): Flattened phase order restores all32 original bit positions exactly.
- `partial_combine` (test): Final integer totals include all valid tile partials and no overflow or premature consumption.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
