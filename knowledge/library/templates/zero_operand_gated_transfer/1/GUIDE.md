# zero_operand_gated_transfer@1: annihilator certified zero fill transfer

## When to consider

When a transfer spans only multiplication coordinates whose other operand is exactly zero, replace that source fetch with a defined zero-filled tile and retain the dense compute/completion schedule. The source groups FP4 activation pairs and pair-major weights so each16-byte fetch can be gated without discarding a nonzero product. Profit depends on exact-zero incidence and memory traffic, not fewer MMAs.

## Computation before and after

Gate opposite-operand memory fetch by an exact multiplication annihilator and zero-fill the unchanged dense compute tile.

For a dense paired-K operand load, inspect both corresponding activation codes. In zero_code_pair mode certify both decoded magnitudes zero; sourceE2M1 byte test(code_byte &0x77)==0 ignores each sign bit only. If certified, issue the original16-byte asynchronous copy with source byte count0, defining the destination tile as zeros. Otherwise fetch all16 source bytes. Keep identical copy commit/wait and dense MMA schedule. Out-of-rangeK tail uses the same zero-fill mechanism only under the declared neutral tail contract. The transformed tile is not an equivalent standalone weight tensor; it is equivalent only in the gated contraction context.

Recover the original computation: Restore original opposite-operand values for gated entries; they multiply decoded-zero activation values, so changing those entries to zeros leaves admitted contraction outputs unchanged.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| gating | compile / software | ["unconditional", "zero_code_pair"] | Use ordinary opposite-operand reads or gate each paired-K transfer from two decoded-zero activation codes. |

## Coupled constraints and scope

Pinned E2M1 activation pair and pair-major weight transfer in SM120 MoEFC2. This reduces source traffic while dense MMA count remains unchanged. It is not2:4 compression, a kept interval, or arbitrary small-value pruning. Input zeros must be exact decoded zeros.

Scope predicate: `p.code_zero_predicate_matches_decoder and p.all_gated_products_zero and p.finite_operand_and_scale_values and p.copy_group_and_consumer_protocol_unchanged`.

- Precondition `pre_annihilator` (semantic): `p.code_zero_predicate_matches_decoder and p.all_gated_products_zero and p.finite_operand_and_scale_values and p.signed_zero_unobserved` — Every changed product is neutral under the numerical contract, including decoded scales; NaN/Inf and observable signed-zero behavior are excluded.
- Precondition `pre_layout` (implementation): `p.transfer_covers_only_matching_gated_coordinates and p.opposite_operand_pair_major_layout` — Each16-byte fetch covers exactly the twoK positions tested, across output channels, with no ungated nonzero products hidden in the fetch.
- Precondition `pre_protocol` (implementation): `p.zero_fill_defines_every_destination_byte and p.copy_group_and_consumer_protocol_unchanged` — Skipped source reads still write a complete zero destination and participate in the original stage/completion protocol.
- Constraint `c_pair` (implementation): `p.k_values_per_gate == 2 and p.transfer_bytes == 16` — Pinned twoK-coordinate gate and16-byte opposite-operand transfer.

## Local example and misuse

For a local instance of `contract`, identify the object `annihilator_certified_zero_fill_transfer` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/zero_operand_gated_transfer@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/zero_operand_gated_transfer@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Same destination stages; opposite operand may need offline pair-major permutation.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `gate_complete` (exhaustive): The gate certifies zero for every activation coordinate represented by a suppressed opposite-operand transfer.
- `layout_pairing` (test): No transfer mixes gated and ungatedK pairs; output-channel packing preserves the original MMA fragment mapping.
- `stage_defined` (test): All bytes are zero-filled and original completion/consumer events remain balanced.
- `numeric_zero` (test): Zero replacement preserves the finite contraction policy including unobserved signed zeros and valid scales.

Zero timesNaN/Inf and observable signed-zero reductions invalidate the neutrality argument. Dropping merely small codes is outside scope.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
