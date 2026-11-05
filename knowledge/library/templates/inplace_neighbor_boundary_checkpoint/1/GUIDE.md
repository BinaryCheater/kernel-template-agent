# inplace_neighbor_boundary_checkpoint@1: inplace neighbor boundary checkpoint

## When to consider

When an in-place adjacent difference reads across CTA boundaries, save just those cross-tile neighbors before launching the output pass. Choose left or right direction; avoid retaining the full input. A boundary after index3 must read original x3 for output4 even if the previous CTA already replaced x3.

## Computation before and after

Snapshot cross-tile input neighbors before any in-place stencil output can overwrite them.

For every tile boundary snapshot the preceding input element for left-neighbor differences, or the following tile first element for right-neighbor differences. Complete the checkpoint pass before any tile writes in-place outputs. Each CTA loads its own tile into local values; internal neighbors come from those captured inputs, cross-tile neighbor comes from checkpoint storage, and the global edge uses the declared endpoint policy. Then store outputs over the original input allocation.

Recover the original computation: Replace checkpoint and captured tile reads by their original pre-write input values, recovering an out-of-place adjacent-neighbor operation.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| neighbor | compile / software | ["left", "right"] | Which adjacent input outside a tile is required. |

## Coupled constraints and scope

One-dimensional adjacent-neighbor operations on disjoint tiles with exact in-place input/output aliasing. Arbitrary overlapping shifted allocations or larger stencil halos need a different checkpoint mapping.

Scope predicate: `(p.tile_partition_exact and p.input_output_alias_only_same_elements) and (p.checkpoint_launch_completes_before_output_writes and p.checkpoint_live_through_last_read) and (p.tile_inputs_loaded_before_local_overwrite and p.endpoint_policy_explicit)`.

- Precondition `pre0` (semantic): `p.tile_partition_exact and p.input_output_alias_only_same_elements` — Tiles cover the input with the admitted in-place aliasing.
- Precondition `pre1` (semantic): `p.checkpoint_launch_completes_before_output_writes and p.checkpoint_live_through_last_read` — A preceding checkpoint pass preserves boundary input values.
- Precondition `pre2` (semantic): `p.tile_inputs_loaded_before_local_overwrite and p.endpoint_policy_explicit` — Within-tile inputs are captured before stores and global endpoints follow the caller rule.

## Local example and misuse

For a local instance of `map`, identify the object `inplace_neighbor_boundary_checkpoint` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cccl/delta/inplace_neighbor_boundary_checkpoint@1/template.yaml`. Original outline: `knowledge/studies/cccl/delta/inplace_neighbor_boundary_checkpoint@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `snapshot_identity` (proof): Each cross-tile neighbor equals the original input before any output overwrite.
- `local_capture` (test): All local inputs used by a tile are captured before its destructive output stores.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
