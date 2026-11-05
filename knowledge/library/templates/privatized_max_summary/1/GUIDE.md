# privatized_max_summary@1: privatized max summary

## When to consider

When many updates contend on the same max-register summary, accumulate independent block-local summaries and merge their completed register maxima into one global summary. Private arrays[0,5,1] and[2,3,1] merge to[2,5,1], not[2,8,2]. Tail item after vector groups must update the same hash/register policy before its owning block merges.

## Computation before and after

When many updates contend on the same max-register summary, accumulate independent block-local summaries and merge their completed register maxima into one global summary.

Initialize a CTA-private register array to zero, synchronize, apply each assigned item update as local atomic max, synchronize, then cooperatively atomic-max every local register into the corresponding global register. If shared allocation/reservation is unavailable, process items with direct global max instead. Vectorized item reads use exactly one tail owner to avoid omissions; repeated items do not change idempotent max semantics.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| local_registers | compile / software | {"lo": 1, "hi": 1073741824} | Summary register count; must fit chosen private shared/global capacity. |

## Coupled constraints and scope

NVIDIA HLL add_if shared/global kernels. CCCL privatized_bin_accumulation@1 counts+1 and sums private counters; it cannot encode max-register flush unchanged. This rule preserves register state and does not introduce a cardinality accuracy claim. Resource effects: Shared summary per CTA plus one global merge per register can reduce global contention but adds full-array clear/flush work; benefit depends on input volume, precision and resident block count.

Scope predicate: `p.input_partition_complete and p.summary_mapping_identical and p.local_global_initial_zero and p.max_updates_atomic and p.local_updates_complete_before_merge and p.global_read_after_merge`.

- Precondition `pre_0` (semantic): `p.input_partition_complete and p.summary_mapping_identical` — Every admitted update is processed and every private/global summary uses the same register/rank mapping.
- Precondition `pre_1` (semantic): `p.local_global_initial_zero and p.max_updates_atomic` — All summaries begin at their max identity and competing register updates are atomic.
- Precondition `pre_2` (semantic): `p.local_updates_complete_before_merge and p.global_read_after_merge` — Local producers finish before flush and final estimate follows every block merge.

## Local example and misuse

For a local instance of `map`, identify the object `privatized_max_summary` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cucollections/delta/privatized_max_summary@1/template.yaml`. Original outline: `knowledge/studies/cucollections/delta/privatized_max_summary@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Shared summary per CTA plus one global merge per register can reduce global contention but adds full-array clear/flush work; benefit depends on input volume, precision and resident block count.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `summary_preservation` (proof): Each local array is the maximum of its assigned updates, and final global is their pointwise maximum.
- `phase_lifetime` (proof): Private initialization precedes updates and merge reads follow all updates.
- `tail_cover` (proof): Vector groups and scalar remainder cover the item domain without missing a remainder.
- `resources` (proof): Local register bytes fit shared allocation; fallback changes placement, not sketch mapping.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
