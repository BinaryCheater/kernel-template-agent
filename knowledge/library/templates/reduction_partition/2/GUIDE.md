# reduction_partition@2: reduction group

## When to consider

Paper GEMM step 1: one sum of K products becomes partial sums over groups followed by a combine. The decision is the assignment of reduction indices to groups; the obligation is exact-once.

## Computation before and after

split one reduction into groups of partial sums followed by a combine

for each output: for r in groups: partial[r] = sum of the products whose index is assigned to r; result = combine(partial[0..num_groups-1]). With contiguous_sequential the partials are summed in place into one accumulator (the k-loop of a tiled GEMM); with contiguous_parallel each group is a separate executor (split-K) and the combine is a separate reduction. With range_policy=balanced_quotient let q=length//num_groups,r=length%num_groups. Split s owns [s*q+min(s,r), s*q+min(s,r)+q+[s<r]). group_size=ceil(length/num_groups) is the maximum group size, not a constant start stride. Empty splits publish identity and complete arrival/turnstile even with no operand loads. Nonlinear epilogue applies only after all partials are combined.

Recover the original computation: Combining the group partial results gives the sum over all reduction indices, because the groups partition the index set; forget the groups.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| num_groups | runtime / software | {"lo": 1, "hi": 1048576} | number of groups (partial results) per output |
| group_size | compile / software | {"lo": 1, "hi": 65536} | reduction indices per group (e.g. the k-tile size) |
| assignment | compile / software | ["contiguous_sequential", "contiguous_parallel", "contiguous_ranges", "strided_threads", "strided_warps", "runtime_balanced"] | which indices go to which group, and whether groups are summed one after another into one accumulator (sequential) or by different executors (parallel) |
| range_policy | compile / software | ["uniform", "balanced_quotient"] | Uniform groups retain version1 mapping, or balance K tiles by quotient/remainder across fixed split participants. |

## Coupled constraints and scope

Reductions whose combine operator is associative and commutative in exact arithmetic (sum, max, min). Floating-point sums qualify only up to rounding; see numeric_risk. Version2 adds balanced quotient ranges and empty participant completion. Version1 uniform nonempty bindings translate with range_policy=uniform and the identity premise vacuous when no group is empty.

Scope predicate: `p.assoc and p.comm and p.empty_groups_emit_identity`.

- Precondition `pre_cover` (semantic): `num_groups * group_size >= p.length` — every product is assigned to a group (exact-once also needs the assignment to be a partition, obligation exact_once)
- Precondition `pre_empty_identity` (semantic): `range_policy != 'balanced_quotient' or p.empty_groups_emit_identity` — Every empty split still publishes neutral partial and arrives at the finalizer protocol.
- Constraint `c_cover` (semantic): `num_groups * group_size >= p.length` — the groups cover every reduction index
- Constraint `c_no_empty` (implementation): `p.empty_groups_emit_identity or (num_groups - 1) * group_size < p.length` — Empty participants are legal only if they emit the true combine identity and complete the same protocol.

## Local example and misuse

For a local instance of `reduce`, identify the object `reduction_group` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/quack/delta/reduction_partition@2/template.yaml`. Original outline: `knowledge/studies/quack/delta/reduction_partition@2/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions:
- id: combine
  pattern: reduce
  kind: ffma
  count: (num_groups - 1) * p.outputs
  amounts:
    ops: '1'
  when: assignment != 'contiguous_sequential'
storage: []
overlap: []
```

## Obligations and numerical checks

- `exact_once` (exhaustive): Every reduction index k is assigned to exactly one group (the assignment is a partition of 0..length-1).
- `combine_complete` (test): The combine reads every group's partial result exactly once and none before it is final.

Changes the summation order, so floating-point results differ from a sequential sum within rounding. Checks must include the input distribution "large_dynamic_range" (values spanning many binades with cancellation).

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- enables: [physical_storage@1](../../physical_storage/1/GUIDE.md); partial results become values that need storage

Structured contract: [template.yaml](template.yaml).
