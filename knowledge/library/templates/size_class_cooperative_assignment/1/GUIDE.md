# size_class_cooperative_assignment@1: size class cooperative assignment

## When to consider

When a batch mixes tiny and large tasks, choose size thresholds and cooperative width rather than giving every task a full CTA. Preserve descriptor identities through partitioning. Memcpy may service tiny buffers locally and queue large ones; segmented reduction can choose a subgroup launch from a safe maximum segment size. Reject queues that lose task identity or lanes that execute another task collective.

## Computation before and after

Independent tasks are partitioned by size and assigned thread-, subgroup- or CTA-cooperative executors.

Classify each task by its actual or safe upper-bound size: thread class up to thread_threshold, subgroup class up to subgroup_threshold, otherwise CTA class. Preserve original task IDs in class worklists. A subgroup lane l handles offsets l+j*subgroup_width; CTA thread l uses l+j*block_threads; a single thread iterates its whole tiny task. Locally service smaller classes and enqueue larger tasks when the source uses a separate kernel; allocate queue slots by exact count prefixes. An alternative uniform launch may choose one class for all tasks using a declared maximum task size. Operator-specific reduction/scan/copy is unchanged inside each task.

Recover the original computation: Erase class queues and executor grouping to recover the original independent per-task computations over all assigned item offsets.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| thread_threshold | compile / software | {"lo": 1, "hi": 65536} | Maximum task size in the thread-cooperative class. |
| subgroup_threshold | compile / software | {"lo": 1, "hi": 65536} | Maximum size in the subgroup-cooperative class. |
| subgroup_width | compile / software | {"lo": 1, "hi": 65536} | Threads collaborating on one middle-class task. |

## Coupled constraints and scope

Independent variable-size tasks with legal cooperative bodies and explicit size units. Classification threshold is a scheduling policy, not a numerical change. Subgroup collectives require their own full-participation contract.

Scope predicate: `(thread_threshold <= subgroup_threshold and p.size_classification_consistent) and (p.task_ids_preserved and p.class_worklists_exact_once and p.cooperative_lane_maps_partition_task) and (p.scratch_and_launch_legal and p.task_outputs_independent)`.

- Precondition `pre0` (semantic): `thread_threshold <= subgroup_threshold and p.size_classification_consistent` — Size classes are ordered, exhaustive and use one size unit.
- Precondition `pre1` (semantic): `p.task_ids_preserved and p.class_worklists_exact_once and p.cooperative_lane_maps_partition_task` — Partitioning and cooperative item maps preserve every original task exactly once.
- Precondition `pre2` (semantic): `p.scratch_and_launch_legal and p.task_outputs_independent` — Cooperative resources/participants are legal and task outputs do not conflict.

## Local example and misuse

For a local instance of `work_assign`, identify the object `size_class_cooperative_assignment` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cccl/delta/size_class_cooperative_assignment@1/template.yaml`. Original outline: `knowledge/studies/cccl/delta/size_class_cooperative_assignment@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `task_cover` (proof): Every task enters one size class and each of its items is assigned to one cooperating lane.
- `queue_and_resources` (test): Class queues, lane masks and per-task scratch match the selected launch.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
