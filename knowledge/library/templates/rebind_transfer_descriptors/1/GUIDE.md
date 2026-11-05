# rebind_transfer_descriptors@1: per worker descriptor generation

## When to consider

When a persistent transfer keeps its tile format but switches problem addresses, shapes or strides, consider rebinding a private descriptor tuple instead of creating another kernel. Preserve the current problem's coordinate mapping and retire old descriptor readers before release publication and reader acquire; bind descriptor_count to the actual publisher tuple. The pinned SMEM-shadow TMA path supports input tuples4 or2 and output tuple1 with128-byte descriptors; K0 still requires its output descriptor, and other mutation/proxy protocols need a variant.

## Computation before and after

Cache a problem key and publish matching private descriptor generations before their transfers.

Initialize private shared descriptor shadows from fixed transfer atoms. For a consumed tuple whose problem key differs from the publisher's cached key, construct tensors with that problem's base addresses/extents/strides and update the shared shadows. Retire all users of the old published generation before publishing new shadows into the publisher's global descriptor slots with the required release operation. Readers acquire every descriptor they will use before issuing the new tuple's transfers. Cache the new problem key. Reusing the same immutable key skips updates. A skipped input transfer does not imply that an output descriptor can be skipped, including K0 output work.

Recover the original computation: Replace each use of a mutable descriptor generation by an immutable descriptor built for the exact problem metadata of that transfer.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| descriptor_count | compile / boundary | {"lo": 1, "hi": 5} | Descriptors owned by one publisher tuple; source SM100 input tuple4, SM103 AB/SF tuples2, output tuple1. |

## Coupled constraints and scope

Per-worker mutable descriptor tuples with shared shadows and release-published global slots while fixed transfer instructions/layouts are reused across problems. Pinned grouped kernels use128-byte TMA descriptors and five total per-CTA slots divided among independent input/output publishers. The source helper's direct- global update branch is not this shared-staging implementation. Different rank, format, field mutation, shared slot ownership or proxy/lifetime semantics needs an explicit variant or newly verified binding. Descriptor traffic/latency is unpriced.

Scope predicate: `p.descriptor_layouts_legal and p.problem_key_metadata_immutable and p.fixed_transfer_fields_compatible and p.descriptor_slots_exclusive and p.old_descriptor_readers_retired and p.release_publish_supported and p.consumer_acquire_before_use`.

- Precondition `pre_key` (semantic): `p.problem_key_metadata_immutable and p.fixed_transfer_fields_compatible` — Equal problem keys imply unchanged metadata; data format and fixed transfer layout remain compatible with the templates.
- Precondition `pre_lifetime` (semantic): `p.descriptor_slots_exclusive and p.old_descriptor_readers_retired` — A publisher owns its slots and cannot overwrite a generation still consumed by an older transfer.
- Precondition `pre_publication` (implementation): `p.release_publish_supported and p.consumer_acquire_before_use` — The selected backend publishes complete descriptor generations and readers acquire the current generation before issue.
- Constraint `c_tuple` (implementation): `descriptor_count == p.publisher_tuple_size and p.descriptor_layouts_legal` — Every selected descriptor has valid rank/alignment/layout and the tuple covers the publisher's transfers.

## Local example and misuse

For a local instance of `transfer`, identify the object `per_worker_descriptor_generation` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/rebind_transfer_descriptors@1/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/rebind_transfer_descriptors@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Shared shadows plus private global descriptor slots.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: descriptor_shadow
  level: smem
  scope: cta
  bytes: descriptor_count * p.descriptor_bytes
- id: descriptor_published
  level: gmem
  scope: cta
  bytes: descriptor_count * p.descriptor_bytes
overlap: []
```

## Obligations and numerical checks

- `coordinate_identity` (test): Each issued descriptor maps transfer coordinates to the current problem's intended address and bounds without mixing tuple generations.
- `descriptor_lifetime` (test): No published descriptor generation is mutated while any earlier transfer can still consume it.
- `publication` (test): Shadow fields are complete before release publication and every issuing reader acquires the current generation.
- `skip_key` (exhaustive): An update is skipped only when the metadata of the consumed tuple is unchanged; input-zero-work and output ownership are distinguished.

Address/metadata refinement only; tensor values and bounds must remain those of the current problem.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
