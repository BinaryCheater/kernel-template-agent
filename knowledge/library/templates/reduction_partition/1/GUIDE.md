# reduction_partition@1: reduction group

## When to consider

A long reduction can be consumed in smaller groups while preserving exact-once input ownership and an admitted floating-point summation order.


## Computation before and after

split one reduction into groups of partial sums followed by a combine

for each output: for r in groups: partial[r] = sum of the products whose index is assigned to r; result = combine(partial[0..num_groups-1]). With contiguous_sequential the partials are summed in place into one accumulator (the k-loop of a tiled GEMM); with contiguous_parallel each group is a separate executor (split-K) and the combine is a separate reduction.

Recover the original computation: Combining the group partial results gives the sum over all reduction indices, because the groups partition the index set; forget the groups.

## Bind the rule

1. Caller fixes p.length and the operator; p.assoc/p.comm describe its admitted exact-arithmetic algebra, not bitwise floating-point equivalence.
2. Choose group_size and assignment. Derive num_groups=ceil(p.length/group_size) for contiguous groups, then check c_cover and c_no_empty.
3. For contiguous_sequential accumulate successive groups into the same owned output. For parallel groups supply a separate correctly synchronized combine and storage for the partials.
4. Guard the last group's loads at p.length and supply the operator's identity for padded positions.


| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| num_groups | runtime / software | {"lo": 1, "hi": 1048576} | number of groups (partial results) per output |
| group_size | compile / software | {"lo": 1, "hi": 65536} | reduction indices per group (e.g. the k-tile size) |
| assignment | compile / software | ["contiguous_sequential", "contiguous_parallel", "contiguous_ranges", "strided_threads", "strided_warps", "runtime_balanced"] | which indices go to which group, and whether groups are summed one after another into one accumulator (sequential) or by different executors (parallel) |

## Coupled constraints and scope

Reductions whose combine operator is associative and commutative in exact arithmetic (sum, max, min). Floating-point sums qualify only up to rounding; see numeric_risk.

Scope predicate: `p.assoc and p.comm`.

- Precondition `pre_cover` (semantic): `num_groups * group_size >= p.length` — every product is assigned to a group (exact-once also needs the assignment to be a partition, obligation exact_once)
- Constraint `c_cover` (semantic): `num_groups * group_size >= p.length` — the groups cover every reduction index
- Constraint `c_no_empty` (implementation): `(num_groups - 1) * group_size < p.length` — no group is empty

## Local example and misuse

K=769, group_size=32, num_groups=25, assignment=contiguous_sequential. Group 24 contains k=768 only; its other 31 slots contribute zero. A split-K implementation cannot reuse this binding unchanged because partial-output ownership and the combine are different.


## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/seed/templates/reduction_partition.yaml`. Original outline: `knowledge/seed/proofs/reduction_partition.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

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
