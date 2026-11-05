# step_pinned_device_page_residency@1: step pinned device page residency

## When to consider

When a step needs only a bounded subset of expert payloads, decide residency on the device, pin every page needed by that step, and replace only unneeded slots before copying the complete payload generation and remapping route IDs.

## Computation before and after

When a step needs only a bounded subset of expert payloads, decide residency on the device, pin every page needed by that step, and replace only unneeded slots before copying the complete payload generation and remapping route IDs.

Advance the device step counter. Mark all already resident experts required by this step with its age. Process missing required IDs in route order; choose the oldest unpinned slot, then smallest slot ID, evict its old inverse map, install the new expert in both maps, pin it at current age, and append one expert/slot copy plan record. A repeated expert is now resident and creates no second copy. Copy every tensor of each planned expert into the same chosen slot using the plan count read on the device. After the complete copy generation is ready, execute routes using resident slot IDs and original route weights. Inactive routes use a safe slot and zero weight. Do not start a subsequent eviction while prior payload readers remain active.

Recover the original computation: For each active original expert route, its remapped slot contains that same expert's complete matching payload generation before use; weights and route multiplicity are unchanged. Unneeded cache pages affect only replacement work, not visible expert results.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| page_slots | runtime / software | {"lo": 1, "hi": 1048576} | Cached complete expert payload slots; source requires total routed entries per step no greater than this capacity. |

## Coupled constraints and scope

Source paged_experts.cuh:27-159 uses one warp for device plan updates, one lane committing each eviction, and mapped pinned-host→GPU gather over all paged tensors. Source guard is routed-entry count<=K, stronger than distinct-working-set capacity. Copy units16/4/2 bytes follow tensor row width and alignment. Counter overflow, concurrent plan writers, incomplete tensor generations and inaccessible host pointers are excluded.

Scope predicate: `(p.route_count <= page_slots and p.initial_slot_maps_mutually_inverse and p.required_ids_valid_or_inactive) and (p.previous_slot_readers_complete and p.one_serial_plan_writer and p.step_counter_monotone_without_wrap) and (p.all_initially_required_residents_pinned_before_eviction and p.newly_installed_pages_pinned_immediately) and (p.victim_is_oldest_unpinned_then_smallest_slot and p.eviction_updates_both_maps) and (p.host_payload_immutable_mapped_and_accessible and p.all_payload_tensors_use_same_plan) and (p.run_consumers_wait_for_complete_payload_generation and p.inactive_routes_have_zero_weight)`.

- Precondition `pre_0` (implementation): `p.route_count <= page_slots and p.initial_slot_maps_mutually_inverse and p.required_ids_valid_or_inactive` — The source capacity bound guarantees room for every required unique expert; slot/expert maps initially agree.
- Precondition `pre_1` (implementation): `p.previous_slot_readers_complete and p.one_serial_plan_writer and p.step_counter_monotone_without_wrap` — No prior step still consumes a replaceable slot; plan updates serialize and current-step age markers cannot alias an older generation.
- Precondition `pre_2` (implementation): `p.all_initially_required_residents_pinned_before_eviction and p.newly_installed_pages_pinned_immediately` — Every current-step resident is protected before victim selection; duplicates of newly installed IDs reuse their slot.
- Precondition `pre_3` (implementation): `p.victim_is_oldest_unpinned_then_smallest_slot and p.eviction_updates_both_maps` — Choose LRU among pages not used by this step, preserving deterministic slot ties and inverse maps.
- Precondition `pre_4` (implementation): `p.host_payload_immutable_mapped_and_accessible and p.all_payload_tensors_use_same_plan` — Every paged weight/scale tensor reads the same expert→slot plan from valid mapped source storage.
- Precondition `pre_5` (implementation): `p.run_consumers_wait_for_complete_payload_generation and p.inactive_routes_have_zero_weight` — Slot assignment alone is not data readiness; readers wait for all required page copies, and inactive routes cannot contribute.

## Local example and misuse

For a local instance of `map`, identify the object `step_pinned_device_page_residency` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/step_pinned_device_page_residency@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/step_pinned_device_page_residency@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Reusable bounded payload slots with explicit generation identity.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `logical_identity` (proof): For each active original expert route, its remapped slot contains that same expert's complete matching payload generation before use; weights and route multiplicity are unchanged. Unneeded cache pages affect only replacement work, not visible expert results.
- `publication_and_lifetime` (test): Native readiness, index maps and memory lifetimes implement the declared complete generation.

If U is the required unique set with|U|<=route_count<=K, every previously required resident is pinned and every newly installed member is pinned. Before the next missing member is placed, fewer than|U|<=K slots are pinned, so an unpinned victim exists. Updating both inverse maps preserves their bijection; duplicates find the new slot. Matching complete payload copies establish result identity. Host checks/page_residency.json covers repeated routes, tie victims and all-needed pinning; native mapped-host bandwidth and synchronization are not claimed.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
