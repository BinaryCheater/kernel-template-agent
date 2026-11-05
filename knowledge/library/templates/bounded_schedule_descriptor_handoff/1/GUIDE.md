# bounded_schedule_descriptor_handoff@1: bounded schedule descriptor handoff

## When to consider

When a collective schedule is complex but payload work is regular, use one scheduler thread to publish bounded descriptors to independent worker groups and retire each descriptor before ring reuse. reduce_scatter.h:95-235: one-RPN flags ready/free; multi-RPN step epoch/-1, local NVLS stage and PAT stage coordinate retirement. src/include/collectives.h:455-718 supplies schedule, not this queue rule.

## Computation before and after

When a collective schedule is complex but payload work is regular, use one scheduler thread to publish bounded descriptors to independent worker groups and retire each descriptor before ring reuse.

Scheduler waits until descriptor slot i%D is retired, writes complete operation indices/dimensions/offsets/flags and release-publishes ready or step epoch. Worker group g consumes steps i=g+k*G with acquire readiness/epoch match, executes transfer/reduce including per-connection credit updates, then release-retires descriptor. Full-step generation is used when multiple stage readers can observe the ring. Retire only after every required reader finishes. Termination markers reach every group; skipped operations retain required handshakes.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| descriptor_slots | compile / software | {"lo": 1, "hi": 4096} | Finite shared descriptor ring depth. |
| worker_groups | compile / software | {"lo": 1, "hi": 32} | Disjoint warp-aligned groups consume steps by stride. |

## Coupled constraints and scope

Source PAT scheduler/worker separation; PAT rank/dimension arithmetic must be bound and proved separately. This rule does not relabel arbitrary work stealing as correct communication. Resource effects: One scheduler thread and bounded shared descriptors; workers overlap schedule construction and payload movement.

Scope predicate: `p.descriptor_sequence_matches_collective_dependencies and p.descriptor_single_writer_and_assigned_reader and p.release_acquire_descriptor_publication_and_retirement and p.worker_group_barriers_disjoint and p.skipped_steps_and_terminal_markers_preserve_protocol and p.resource_budget_legal`.

- Precondition `pre_0` (semantic): `p.descriptor_sequence_matches_collective_dependencies and p.descriptor_single_writer_and_assigned_reader` — Schedule legality is a separate premise, not proved by queue synchronization.
- Precondition `pre_1` (semantic): `p.release_acquire_descriptor_publication_and_retirement and p.worker_group_barriers_disjoint` — Payload and metadata fully read before return to scheduler; selected descriptor epoch matches.
- Precondition `pre_2` (semantic): `p.skipped_steps_and_terminal_markers_preserve_protocol and p.resource_budget_legal` — Skipped data work may still consume/retire protocol slots; all active groups receive termination.

## Local example and misuse

For a local instance of `map`, identify the object `bounded_schedule_descriptor_handoff` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/nccl/delta/bounded_schedule_descriptor_handoff@1/template.yaml`. Original outline: `knowledge/studies/nccl/delta/bounded_schedule_descriptor_handoff@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One scheduler thread and bounded shared descriptors; workers overlap schedule construction and payload movement.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `publication` (proof): Acquired ready/epoch exposes all fields from exactly that descriptor generation.
- `bounded_reuse` (proof): Scheduler cannot overwrite a descriptor while any assigned reader needs it.
- `schedule_separation` (proof): Handoff preserves declared operation order/dependencies but does not establish correctness of the underlying rank schedule.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
