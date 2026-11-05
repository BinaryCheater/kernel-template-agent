# cost_partitioned_worklists@1: cost planned per worker descriptor lists

## When to consider

When independent ragged tasks have unequal estimated costs and their sizes are known before launch, preassign them to least-loaded CTA lists. Flatten complete descriptors and have each CTA walk only its prefix-offset range. This replaces global fetch/stride scheduling with an immutable plan, trading planning and metadata traffic for controlled assignment. Costs select placement; exact cover and complete metadata preserve semantics, with no promise that the estimate predicts runtime.

## Computation before and after

Greedy least-estimated-load task partition followed by per-CTA contiguous descriptor-list traversal.

Given an ordered immutable task enumeration, initialize each worker load to zero and a min-heap of (estimated accumulated load, worker identity). For each task, remove a least-loaded worker, append the complete task descriptor to that worker list, add the task estimated cost, and reinsert the worker. Keep the chosen tie policy defined; exact ties need not have a source-independent order. Flatten worker lists in worker-ID order, permuting all metadata fields together, and emit work_indptr[0..workers]. Launch one CTA per worker. CTA b executes descriptor positions work_indptr[b] through work_indptr[b+1]-1 in order; every role advances to the same next descriptor only after old task state is safe to reuse. Empty ranges execute no task. Optional replicated-head schedules add the actual grid head to each descriptor identity. Cost affects assignment, never task semantics or coverage; changing costs requires rebuilding the plan before its consumers launch.

Recover the original computation: Unroll each worker range into its original complete task descriptors and discard worker assignment/costs; this reconstructs the original task multiset exactly once.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| workers | runtime / software | {"lo": 1, "hi": 4096} | Number of planned worker lists and CTA workers per grid slice; explicit head replication may launch several slices sharing the same lists. |

## Coupled constraints and scope

Host-preplanned immutable lists of independent CTA tasks, with finite nonnegative estimated costs. Source SM90 prefill sorts requests by decreasing KV length and visits query tiles in reverse before least-load assignment. The task-order and cost functions are caller bindings, not guarantees of optimal load or measured speedup. No runtime stealing, splitting of individual tasks, or cross-CTA dependency repair is supplied.

Scope predicate: `p.ordered_tasks_exact_cover_original_work and p.tasks_independent_or_dependencies_preserved and p.task_costs_finite_nonnegative_and_heap_comparison_defined and p.descriptors_retain_all_original_task_coordinates_and_generations and p.parallel_descriptor_fields_permuted_together and p.flattened_lists_fit_descriptor_capacity_and_index_type and p.work_indptr_monotone_with_correct_endpoints and p.shared_plan_replication_preserves_distinct_task_identity_and_coverage and p.complete_plan_published_before_grid_and_immutable_until_all_readers_retire and p.worker_roles_advance_same_list_position_and_finish_old_task_before_state_reuse and p.selected_body_and_grid_resources_legal`.

- Precondition `pre_tasks` (semantic): `p.ordered_tasks_exact_cover_original_work and p.tasks_independent_or_dependencies_preserved` — The ordered task list covers original work once; assigning different workers preserves dependencies and output ownership.
- Precondition `pre_cost` (implementation): `p.task_costs_finite_nonnegative_and_heap_comparison_defined` — Each estimated cost is finite and nonnegative; all heap comparisons and tie choices are defined.
- Precondition `pre_descriptors` (semantic): `p.descriptors_retain_all_original_task_coordinates_and_generations and p.parallel_descriptor_fields_permuted_together` — Every descriptor carries complete task identity and all parallel metadata arrays move together.
- Precondition `pre_ranges` (implementation): `p.flattened_lists_fit_descriptor_capacity_and_index_type and p.work_indptr_monotone_with_correct_endpoints` — The flattened lists and worker prefix offsets fit their allocated buffers and integer types.
- Precondition `pre_replay` (semantic): `p.shared_plan_replication_preserves_distinct_task_identity_and_coverage` — Any head or other grid-axis replication restores that axis in task identity; it neither repeats nor drops original jobs.
- Precondition `pre_lifetime` (implementation): `p.complete_plan_published_before_grid_and_immutable_until_all_readers_retire` — Descriptor generations publish before consumption and survive every worker iteration.
- Precondition `pre_body` (implementation): `p.worker_roles_advance_same_list_position_and_finish_old_task_before_state_reuse and p.selected_body_and_grid_resources_legal` — All roles agree on each task and safely reset or retire per-task state; launch and body resources are legal.
- Constraint `c_workers` (implementation): `workers >= 1 and workers <= p.max_legal_worker_ctas` — Launch one CTA per planned worker within the supported grid/resource bound.
- Constraint `c_capacity` (resource): `p.task_count >= 0 and p.descriptor_capacity >= p.task_count and p.offset_capacity >= workers + 1` — The plan has space for every task and all worker prefix offsets.

## Local example and misuse

For a local instance of `work_assign`, identify the object `cost_planned_per_worker_descriptor_lists` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/cost_partitioned_worklists@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/cost_partitioned_worklists@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: A plan can be reused only while workload identities, sizes, shape and launch bindings remain valid.. Storage: Full task descriptors plus worker prefix offsets in global memory; temporary planner lists and heap.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: worklists
  level: dram
  scope: kernel
  bytes: p.task_count * p.descriptor_bytes + (workers + 1) * p.index_bytes
overlap: []
```

## Obligations and numerical checks

- `partition_cover` (proof): Appending every original task once and flattening disjoint worker lists preserves exact task coverage.
- `metadata_and_replication` (test): All descriptor fields retain matching task identities and generations; any grid replication restores the omitted axis exactly once.
- `range_and_lifetime` (test): Prefix endpoints and validity tests prevent out-of-range reads; plan publication and storage outlive all iterations.
- `worker_iteration` (test): All roles consume the same task and retire/reset its pipeline and accumulator state before advancing.

Independent task arithmetic is unchanged only when per-task accumulators and pipeline state are reset and no hidden order-dependent cross-task side effects exist.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
