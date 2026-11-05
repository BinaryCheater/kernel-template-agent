# independent_operation_coalescing@1: independent operation coalescing

## When to consider

When independent row tasks share a launch boundary, place their task domains in one kernel as disjoint CTA ranges or sequence both row tasks inside one CTA, preserving each body and its own dependency waits.

## Computation before and after

When independent row tasks share a launch boundary, place their task domains in one kernel as disjoint CTA ranges or sequence both row tasks inside one CTA, preserving each body and its own dependency waits.

Start with two independent task kinds A(row),B(row),row0..M-1. In disjoint_cta_ranges launch2*M CTAs: indices[0,M) execute A(index),[M,2M) execute B(index-M). In paired_row_sequence launch M CTAs and execute A(row),B(row) in paired_order. Both use block_threads and supported uniform collectives; private scratch may be shared only after its original last read. Each task retains its input producer wait and its output visibility contract. Consumers wait for every task kind they require. A launch/dispatch reduction is possible but no occupancy or overlap speedup follows from this structural transformation.

Recover the original computation: Split each coalesced task instance back into its original kernel/CTA domain. Independence permits the observed interleaving or per-row order without changing any required output; exact task coverage recovers the two original operations.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| task_assignment | compile / software | ["disjoint_cta_ranges", "paired_row_sequence"] | Partition CTAs by task kind, or let one row CTA execute both independent row-task kinds. |
| paired_order | compile / software | ["first_then_second", "second_then_first"] | Execution order of independent row tasks in paired mode; irrelevant for disjoint ranges. |
| block_threads | compile / software | {"lo": 32, "hi": 1024} | One common block shape, legal only when every selected body supports it. |

## Coupled constraints and scope

Source route_quant_fused.cuh uses disjoint ranges: router/pack M CTAs and MXFP8 quant M CTAs,224 threads,3584 input features,group32/two lanes per group. Fused front epilogue in route_radix.cuh uses paired rows with224/448 threads and selectable latent-cast-before/after routing. Small-batch dispatch is a heuristic; PDL-enabled variants retain separate native completion proof, while ordinary kernel-completion semantics give the direct structural instance.

Scope predicate: `(p.two_task_kinds_have_same_row_count and p.no_cross_task_read_write_or_output_alias_hazards) and (p.bodies_support_common_block_and_compilation_policy and p.branch_resource_occupancy_fits) and (p.each_logical_task_assigned_exact_once and p.block_collectives_uniform_inside_selected_task) and (p.every_task_waits_for_its_actual_input_producer and p.external_readers_wait_for_all_required_task_outputs) and (p.same_body_arithmetic_and_required_outputs_preserved)`.

- Precondition `pre_0` (implementation): `p.two_task_kinds_have_same_row_count and p.no_cross_task_read_write_or_output_alias_hazards` — Task instances share a row domain but their reads/writes have no inter-task dependency or destructive alias.
- Precondition `pre_1` (implementation): `p.bodies_support_common_block_and_compilation_policy and p.branch_resource_occupancy_fits` — Both bodies retain legal lane mappings, precision flags, barriers and resource capacity at the common block shape.
- Precondition `pre_2` (implementation): `p.each_logical_task_assigned_exact_once and p.block_collectives_uniform_inside_selected_task` — Task-range or row-sequence mapping covers each logical task once; barriers never mix different CTA branches or divergent participants.
- Precondition `pre_3` (implementation): `p.every_task_waits_for_its_actual_input_producer and p.external_readers_wait_for_all_required_task_outputs` — Each branch carries necessary input readiness and downstream completion; an early launch trigger alone does not publish all outputs.
- Precondition `pre_4` (implementation): `p.same_body_arithmetic_and_required_outputs_preserved` — Both fused task bodies preserve original casts, numeric policy, random identity and all externally required outputs.

## Local example and misuse

For a local instance of `map`, identify the object `independent_operation_coalescing` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/independent_operation_coalescing@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/independent_operation_coalescing@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Shared input with separate logical output regions and explicit temporary lifetimes.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `logical_equivalence` (proof): Split each coalesced task instance back into its original kernel/CTA domain. Independence permits the observed interleaving or per-row order without changing any required output; exact task coverage recovers the two original operations.
- `native_admission` (test): The selected native resource, arithmetic and publication recipe preserves all declared inputs/outputs.

The range maps index→(kind,row) are bijections; paired mode instead executes exactly two independent bodies per row. Disjoint outputs and immutable common inputs allow interleaving without dataflow changes. Full block uniform branches retain barriers. Native common-shape register/shared-memory occupancy and PDL memory ordering must be checked separately; source reuse of a body does not alone certify those resource or early-publication conditions.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
