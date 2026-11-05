# mixed_radix_materialization@1: mixed radix materialization

## When to consider

When a consumer needs contiguous tiles or instruction-owned fragments that ordinary strides cannot express, materialize a mixed-radix digit permutation. Bind the split factors and exact order, preserving code identities and excluding padding. Hopper fragment/scale profiles and Blackwell pre-transposed weight tiles differ in their digit maps; they are not interchangeable swizzles.

## Computation before and after

Split logical coordinates into bounded digits, permute the digits, and materialize a consumer-oriented tensor layout.

Decode coordinates into the selected profile digits, permute those digits, then flatten using destination strides. square_tile splits [R/t,t,C/t,t] and permutes [0,2,1,3]. blackwell_weight_tile takes K-packed bytes and maps (e,k,n) to (e,k/tile_rows,n/tile_columns,n%tile_columns,k%tile_rows); zero-pad tiles. hopper_scale splits [M/(32*warps),2,warps,2,8,K/2,2] and permutes [0,2,5,1,4,6,3], producing [M/32,K*32]. hopper_mma profiles split each of two axes into [rest,k_tile,warp_tile,threads,scott,contig]. Bind k_tile=[1,4/u], warp_tile=[2,2], threads=[4,4], scott=[2,1], contig=[1,u], with u=4 for MMA v2 and u=1 for v3; permute the resulting12 digits by [0,3,6,10,4,9,7,1,8,2,5,11]. The Hopper bit recoding is a separate reversible step. fp4_axis_repack transposes each2x2 logical nibble tile: from adjacent bytes a,b storing one packed axis produce lo=(a&15)|((b&15)<<4), hi=(a>>4)|(b&240), placing lo/hi at adjacent coordinates of the old packed axis; this map is its own inverse. Inverse materialization applies the inverse permutation and removes only padding.

Recover the original computation: Apply the inverse digit permutation and flatten original logical coordinates, preserving each element or packed-code identity.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| profile | compile / software | ["square_tile", "hopper_mma_v2", "hopper_mma_v3", "hopper_scale", "blackwell_weight_tile", "fp4_axis_repack"] | Pinned factorization/permutation family; all profiles are actual NVIDIA storage paths. |
| tile_rows | compile / software | {"lo": 1, "hi": 65536} | Square tile width, or Blackwell packed-K tile extent. |
| tile_columns | compile / software | {"lo": 1, "hi": 65536} | Blackwell N tile extent. |
| warps | compile / software | {"lo": 1, "hi": 65536} | Hopper scale warp count; power of two. |

## Coupled constraints and scope

Materialized complete mixed-radix profiles from shipped NVIDIA tensor layouts. This moves data; tensor_axis_permutation@1 is only a rank-three view and does not supply these digit factorizations. Warp count and tile extents must satisfy consumer alignment and source guards.

Scope predicate: `(p.logical_extents_fit_factorization_or_neutral_padding and p.selected_profile_matches_consumer) and (p.destination_owned_once and p.padding_excluded_or_neutral and p.element_codes_unchanged) and (p.packing_axis_and_leading_strides_explicit)`.

- Precondition `pre0` (semantic): `p.logical_extents_fit_factorization_or_neutral_padding and p.selected_profile_matches_consumer` — Every valid coordinate has a unique complete mixed-radix expansion compatible with the consumer.
- Precondition `pre1` (semantic): `p.destination_owned_once and p.padding_excluded_or_neutral and p.element_codes_unchanged` — Materialization copies exact element/code identities without races; padding cannot become logical data.
- Precondition `pre2` (semantic): `p.packing_axis_and_leading_strides_explicit` — Packed bytes and logical scalar axes are distinguished and batch addresses preserve actual strides.

## Local example and misuse

For a local instance of `store`, identify the object `mixed_radix_materialization` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/triton/delta/mixed_radix_materialization@1/template.yaml`. Original outline: `knowledge/studies/triton/delta/mixed_radix_materialization@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `digit_bijection` (test): The selected split product equals the padded extents and its permutation is bijective on all valid coordinates.
- `padding_and_stride` (test): Leading strides, packing axes and valid tails remain consistent in both directions.
- `consumer_binding` (test): Selected physical layout matches actual instruction/TMA/decode coordinate use.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
