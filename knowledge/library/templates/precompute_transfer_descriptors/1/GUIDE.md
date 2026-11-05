# precompute_transfer_descriptors@1: immutable problem descriptor table

## When to consider

When many workers reuse transfer metadata for a finite set of problems known before execution, consider a separate setup kernel and immutable descriptor table. Store each key/slot once, publish before consumers and keep it live until all transfers finish. Forward MoE uses one C slot per expert; backward uses AB slots. Descriptor-relative coordinates must not add the encoded token offset again. This differs from rebinding private worker slots: setup costs one launch and global storage grows with problems. Native zero-extent and descriptor proxy visibility still need a supported backend binding.

## Computation before and after

Materialize immutable transfer descriptors by problem key before persistent consumption.

Allocate problems times descriptors_per_problem aligned descriptor slots. Launch one setup kernel before consumers, with disjoint problem-key writers. Construct each selected descriptor from that key's full address/extent/stride metadata and store it at128*(key*descriptors_per_problem+slot). Establish setup completion and required descriptor proxy visibility before transfers begin. Consumers select the immutable tuple by key and use coordinates relative to its encoded base. Leave other operands on their unchanged global descriptors. Keep this table immutable until all consumer transfers finish, and retire the previous invocation before rebuilding it. No runtime key cache or per-worker descriptor mutation is introduced.

Recover the original computation: Replace table lookup by the same descriptor freshly constructed from the immutable metadata at each transfer use.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| descriptors_per_problem | compile / boundary | {"lo": 1, "hi": 5} | Number of selected transfer descriptors stored for each problem key. |

## Coupled constraints and scope

Separate device setup for immutable per-problem transfer descriptors in128-byte slots. The dense MoE source builds one C descriptor per forward expert or two AB descriptors per backward expert; unchanged operands keep global descriptors. Repeated keys share the table without rebinding. Zero extents require backend support or a skipped unused slot. The selected source CLI forces setup on; its unsupported fused/inline route cannot satisfy setup completion. Other descriptor formats, live metadata mutation or overlapping table rebuilds need another binding or variant. Native construction/proxy semantics remain open.

Scope predicate: `p.table_fits_allocation and p.descriptor_layouts_legal and p.metadata_available_before_setup and p.metadata_immutable_through_use and p.setup_writers_disjoint and p.previous_invocation_readers_retired and p.setup_completed_before_consumers and p.descriptor_proxy_visibility_supported and p.consumer_uses_matching_key_and_coordinates`.

- Precondition `pre_metadata` (semantic): `p.metadata_available_before_setup and p.metadata_immutable_through_use` — Every address/extent/stride and selected key remains fixed from setup through last use.
- Precondition `pre_ownership` (semantic): `p.setup_writers_disjoint and p.previous_invocation_readers_retired` — Each table slot has one setup owner and storage is not reused while old readers remain.
- Precondition `pre_publication` (implementation): `p.setup_completed_before_consumers and p.descriptor_proxy_visibility_supported` — Setup completion and the backend's required descriptor visibility precede the first transfer.
- Precondition `pre_coordinates` (semantic): `p.consumer_uses_matching_key_and_coordinates` — Descriptor-relative coordinates must not reapply the base offset already encoded by setup.
- Constraint `c_extent` (implementation): `p.problems >= 1 and p.descriptor_bytes == 128 and p.table_fits_allocation` — The selected source uses a nonempty problem table and aligned128-byte slots with sufficient allocation.
- Constraint `c_tuple` (implementation): `descriptors_per_problem == p.selected_tuple_size and p.descriptor_layouts_legal` — Materialized fields and rank/layout support every selected transfer.

## Local example and misuse

For a local instance of `transfer`, identify the object `immutable_problem_descriptor_table` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/precompute_transfer_descriptors@1/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/precompute_transfer_descriptors@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Problem-scaled global descriptor table.. Structured effects describe work and resource changes, not timings.

```yaml
transactions:
- id: setup
  pattern: launch
  kind: kernel_launch
  count: '1'
  note: One setup launch per table construction; no latency claim.
storage:
- id: descriptor_table
  level: gmem
  scope: device
  bytes: p.problems * descriptors_per_problem * p.descriptor_bytes
  note: Shared immutable table; storage scales with problems rather than persistent
    workers.
overlap: []
```

## Obligations and numerical checks

- `table_injective` (exhaustive): Every selected key/slot owns a unique aligned address inside allocation.
- `coordinate_identity` (test): Every transfer uses the intended problem address and bounds with each base offset applied exactly once.
- `publication_lifetime` (test): All table fields are visible before first use and remain immutable through last use including across invocation reuse.

Metadata refinement only; wrong keys or double offsets change values and group boundaries.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- alternative_of: [rebind_transfer_descriptors@1](../../rebind_transfer_descriptors/1/GUIDE.md), [rebind_transfer_descriptors@2](../../rebind_transfer_descriptors/2/GUIDE.md), [rebind_transfer_descriptors@3](../../rebind_transfer_descriptors/3/GUIDE.md); Immutable key-indexed storage trades problem-scaled table/setup for per-worker mutable generations.

Structured contract: [template.yaml](template.yaml).
