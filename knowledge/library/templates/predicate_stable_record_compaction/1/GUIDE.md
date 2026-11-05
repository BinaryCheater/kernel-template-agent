# predicate_stable_record_compaction@1: predicate stable record compaction

## When to consider

When only a sparse subset of projected primitives is valid, compute block survivor counts and a global prefix, then emit complete records with preserved source identity into compact output. Flags[1,0,1] in block0 and[0,1] in block1 give counts2,1,prefix2,3; original source IDs0,2,4 emit to slots0,1,2. Writing camera ID by original index while other fields use compact index corrupts records.

## Computation before and after

When only a sparse subset of projected primitives is valid, compute block survivor counts and a global prefix, then emit complete records with preserved source identity into compact output.

Pass1 assigns one lane to each input record, evaluates full validity and block-reduces Boolean survivor flags. Inclusive-scan block counts in source block order. Pass2 repeats validity and exclusive-scans flags within the block; surviving lane writes all record fields at previous-block-prefix+local-prefix. Preserve batch/camera/primitive IDs along with projected attributes. Derive row indptr from block prefixes and append total-survivor sentinel. Invalid/tail records reserve no output and do not make partial records.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| records_per_block | compile / software | [256] | Observed packed projection block width; record order follows original block/lane order. |

## Coupled constraints and scope

NVIDIA packed EWA/2DGS projection output. Distinct from prefix-only whole-work filtering and from first-occurrence duplicate election; arbitrary per-record validity becomes a stable compact sequence. Resource effects: Two projection/validity evaluations plus count scan, compact attributes/IDs and row offsets; reduces downstream work/storage when few batch-camera-Gaussian combinations survive.

Scope predicate: `p.input_and_validity_immutable and p.count_emit_same_predicate and p.prefix_sum_no_overflow and p.output_capacity_exact and p.record_fields_same_owner and p.scan_collective_uniform`.

- Precondition `pre_0` (semantic): `p.input_and_validity_immutable and p.count_emit_same_predicate` — Both passes evaluate identical source data and validity logic, including projection/culling numerical branches.
- Precondition `pre_1` (semantic): `p.prefix_sum_no_overflow and p.output_capacity_exact` — Global prefix and record counts fit their integer types and output allocations cover the survivor total.
- Precondition `pre_2` (semantic): `p.record_fields_same_owner and p.scan_collective_uniform` — All fields of a surviving record use the same offset; every block participant reaches scans/barriers correctly.

## Local example and misuse

For a local instance of `map`, identify the object `predicate_stable_record_compaction` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/gsplat/delta/predicate_stable_record_compaction@1/template.yaml`. Original outline: `knowledge/studies/gsplat/delta/predicate_stable_record_compaction@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Two projection/validity evaluations plus count scan, compact attributes/IDs and row offsets; reduces downstream work/storage when few batch-camera-Gaussian combinations survive.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `survivor_identity` (proof): Each admitted source record emits once and rejected records emit none.
- `record_atomicity` (proof): All fields share one source owner and compact index; no projection field is paired with another primitive ID.
- `stable_offsets` (proof): Block/global scan composition yields disjoint source-order survivor indices and correct row boundaries.
- `validity_replay` (proof): Floating validity decisions agree in both passes for unchanged inputs and selected specialization.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
