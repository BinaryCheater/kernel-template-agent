# segmented_axis_work_decode@1: segmented axis work origin

## When to consider

When differently sized groups share one contiguous tensor axis, consider decoding separately rounded per-group work rather than tiling the fused extent. Preserve the physical prefix for origins and return each member's signed distance to its group end; choose tile_extent and bind cooperating_tiles to the actual execution unit. The pinned N-axis scalar cursor skips empty groups but keeps inactive cluster members participating; using global tile origins or dropping a peer requires a different mapping/protocol.

## Computation before and after

Inverse separately rounded segment work counts plus unpadded physical origins.

Let unit_extent=tile_extent*cooperating_tiles and count[g]=ceil_div(prefix[g+1]- prefix[g],unit_extent). Invert the inclusive prefix of those counts using a retained forward scalar cursor; empty segments contribute zero work. A valid query returns group g and local unit index u. Member j has physical origin prefix[g]+u*unit_extent+j*tile_extent and signed remaining=prefix[g+1]-origin. Keep the group's work identity for every member, including remaining<=0. Only local coordinates0<=x<tile_extent and x<remaining may store a visible element. Invalid past-total queries publish a terminal identity and do not consume nonexistent group metadata. Do not replace per-group rounding with rounding the total physical extent.

Recover the original computation: Expand all segments' visible coordinates in input order; each valid output belongs to exactly one local unit/member/coordinate and its original physical prefix offset.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| tile_extent | compile / software | {"lo": 1, "hi": 65536} | Extent of one member tile along the segmented tensor axis. |
| cooperating_tiles | compile / boundary | {"lo": 1, "hi": 16} | Member tiles covering one cooperative work unit along that axis; the pinned source uses cluster N. |

## Coupled constraints and scope

One physically contiguous segmented tensor axis with independently rounded cooperative tile work, a monotone scalar search and explicit per-member tail bounds. Pinned mixed-input grouped GEMM segments the fused N axis while A and scale batch select the returned group. The last CTA may have zero/negative remaining distance but still participates in collective pipelines. A different segmentation, backwards query order, integer width or cooperative participation policy needs a variant or verified binding. Physical cursor/register costs and native clipping/arrival progress remain unpriced backend obligations.

Scope predicate: `p.work_unit_layout_legal and p.prefix_and_work_counts_fit_int32 and p.prefix_starts_zero and p.prefix_nondecreasing and p.prefix_complete and p.prefix_metadata_immutable and p.queries_nonnegative_and_monotone and p.consumers_mask_by_segment_end and p.inactive_peer_loads_safe and p.peers_keep_same_work_identity`.

- Precondition `pre_prefix` (semantic): `p.prefix_starts_zero and p.prefix_nondecreasing and p.prefix_complete and p.prefix_metadata_immutable` — One immutable prefix array describes all segment lengths including empty segments.
- Precondition `pre_query` (semantic): `p.queries_nonnegative_and_monotone` — Retained forward cursor never needs an earlier segment.
- Precondition `pre_consume` (implementation): `p.consumers_mask_by_segment_end and p.inactive_peer_loads_safe and p.peers_keep_same_work_identity` — Stores honor per-segment bounds while inactive peers retain safe loads and the cooperative execution sequence.
- Constraint `c_backend` (implementation): `p.work_unit_layout_legal and p.prefix_and_work_counts_fit_int32` — Selected cooperative layout is legal and both physical prefixes and rounded work counts fit the pinned integer representation.

## Local example and misuse

For a local instance of `work_assign`, identify the object `segmented_axis_work_origin` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/segmented_axis_work_decode@1/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/segmented_axis_work_decode@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Immutable prefix array and retained six-field scalar search cursor.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `inverse_rounded_counts` (exhaustive): The cursor returns the unique segment/local unit in the concatenation of separately rounded group counts.
- `physical_origin` (exhaustive): Every member origin and signed remaining distance use that segment's original physical prefix rather than a padded work prefix.
- `visible_coverage` (exhaustive): Per-member boundary masks cover every visible segment coordinate once and no coordinate in a neighboring segment.
- `terminal_and_cooperation` (test): Empty/terminal groups do not read invalid metadata; inactive members do not diverge from required cooperative participation.

Scheduling/map refinement only; storing padded work into a neighboring segment changes visible results.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
