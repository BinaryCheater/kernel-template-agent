# counted_overlap_depth_binning@1: counted overlap depth binning

## When to consider

When one primitive contributes to several screen tiles, duplicate its identity into disjoint prefix-reserved ranges and sort composite tile/depth keys for ordered tile consumers. Counts [2,0,1] give inclusive prefix [2,2,3], emit ranges [0,2),[2,2),[2,3). A Gaussian crossing two tiles emits two records with the same Gaussian ID; deduplicating that ID globally removes a required tile contribution. Negative/NaN depth cannot use this unsigned-bit ordering unchanged.

## Computation before and after

When one primitive contributes to several screen tiles, duplicate its identity into disjoint prefix-reserved ranges and sort composite tile/depth keys for ordered tile consumers.

Count accepted tile intersections for each primitive using the selected bounding-box or ellipse support predicate. Inclusive-scan counts; primitive i emits exactly its count records starting at prefix[i-1] or zero for i=0. Each record pairs original primitive ID with key=(image,tile,positive-depth-bits). Sort paired keys/IDs by tile then depth. Construct each tile range from changes of key prefix, or lower_bound each active tile prefix and append a terminal total-count sentinel. Consumers read each duplicate only for its tile.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| tile_size | compile / software | {"lo": 1, "hi": 1024} | Screen tile side in pixels; legality also constrained by consumer launch/storage. |
| depth_bits | compile / software | [32] | Source float depth occupies low 32 bits of composite key. |

## Coupled constraints and scope

NVIDIA CUDA tile-intersection preprocessing for Gaussian splatting. Rectangular and ellipse overlap are separate support predicates. Sparse active-tile output must include every emitted tile. Equal depths do not establish a stable primitive-ID order unless the downstream contract supplies it. Resource effects: Two overlap passes plus scan and paired radix sort; O(total_intersections) key/ID storage, avoiding all-primitives scans per tile. Sparse range output uses active_tiles+1 offsets.

Scope predicate: `p.count_emit_same_predicate and p.prefix_sum_exact and p.positive_finite_depth and p.key_fields_fit and p.consumer_order_required and p.valid_primitive_ids`.

- Precondition `pre_0` (semantic): `p.count_emit_same_predicate and p.prefix_sum_exact` — Count and emit use identical overlap and clipping predicates; integer scan does not overflow.
- Precondition `pre_1` (semantic): `p.positive_finite_depth and p.key_fields_fit` — Unsigned float bits preserve positive finite depth order and composite image/tile fields fit int64.
- Precondition `pre_2` (semantic): `p.consumer_order_required and p.valid_primitive_ids` — Consumer needs front-to-back order and duplicate records preserve original primitive identity.

## Local example and misuse

For a local instance of `map`, identify the object `counted_overlap_depth_binning` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/gsplat/delta/counted_overlap_depth_binning@1/template.yaml`. Original outline: `knowledge/studies/gsplat/delta/counted_overlap_depth_binning@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Two overlap passes plus scan and paired radix sort; O(total_intersections) key/ID storage, avoiding all-primitives scans per tile. Sparse range output uses active_tiles+1 offsets.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `count_emit` (proof): Same predicate and traversal produce exactly the reserved count, including clipping and zero-radius paths.
- `identity_pairing` (proof): Sorting permutes keys and IDs together and each accepted primitive/tile pair appears exactly once.
- `ordering` (proof): Key encoding preserves image/tile grouping and positive finite depth order without field overlap.
- `range_recovery` (proof): Each tile range is the exact contiguous sorted run, including empty tiles and final sentinel.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
