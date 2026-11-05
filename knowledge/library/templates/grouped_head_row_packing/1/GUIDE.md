# grouped_head_row_packing@1: packed query head row view

## When to consider

When grouped-query heads reuse one KV head but query rows are too few, pack each token's within-group heads into the row dimension. Row r=t*g+u retains token t=r//g and head kv*g+u. For g3,token2,head7, use KV group2,row7. Remap masks/LSE/gradients too. Physical head padding must not wrap a fake head into the next token through modulo decoding. A rank-three axis permutation alone cannot express this split-and-fold mapping.

## Computation before and after

Fold the within-KV-group query-head index into sequence rows while preserving logical token/head semantics.

For token t and query head h, let kv=h//group_size and u=h modulo group_size. Use packed row r=t*group_size+u and KV-head coordinate kv. Keep base pointer and other dimensions unchanged. Represent the row axis hierarchically as (u,t), with strides (source_head_stride,source_token_stride), and the remaining head axis with stride group_size*source_head_stride. Decode t=r//group_size and h=kv*group_size+(r modulo group_size) at every semantic consumer. Apply the corresponding view separately to output, LSE and derivative tensors, whose head-axis positions/strides need not equal Q's. Tile packed rows only after this coordinate contract is established; physical TMA rank workarounds, head padding and materialized copies are distinct implementation bindings.

Recover the original computation: Decode each packed row to its unique original token/head pair and restore the original extents and strides. The same element address and logical metadata identity are recovered.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| group_size | compile / boundary | {"lo": 1, "hi": 128} | Query heads sharing each KV head; determined by the workload, not chosen independently of it. |

## Coupled constraints and scope

Equal complete query-head groups with one through128 heads per KV head and a stride-preserving hierarchical view. This does not admit padded head lanes through ordinary modulo decoding, implicit contiguous reshapes, independently mixing KV groups or assuming a Q head-axis position for LSE. Sparse MLA's dynamic-head TMA padding requires an additional guarded implementation.

Scope predicate: `p.grouped_heads_share_kv and p.source_shape_strides_retained and p.mask_and_score_mod_decode_original_token_head and p.output_and_statistics_use_matching_packing and p.consumer_accepts_hierarchical_packed_view and p.allocation_lifetime_and_write_ownership_valid and p.kv_gradient_reductions_include_all_grouped_heads`.

- Precondition `pre_kv` (semantic): `p.grouped_heads_share_kv and p.source_shape_strides_retained` — All grouped query heads use the same KV head and the original allocation/strides remain available.
- Precondition `pre_coordinates` (semantic): `p.mask_and_score_mod_decode_original_token_head and p.output_and_statistics_use_matching_packing` — Packed rows preserve original token/head coordinates in masks, score callbacks, O, LSE and derivative metadata.
- Precondition `pre_layout` (implementation): `p.consumer_accepts_hierarchical_packed_view and p.allocation_lifetime_and_write_ownership_valid` — The backend accepts the packed access layout; all accesses and writes remain in bounds with correct ownership.
- Precondition `pre_gradient` (semantic): `p.kv_gradient_reductions_include_all_grouped_heads` — When gradients are requested, K/V contributions sum over every original head in the group; a view change cannot discard that reduction.
- Constraint `c_group` (semantic): `p.query_heads == group_size * p.kv_heads` — This version uses complete equal groups without padded logical heads.

## Local example and misuse

For a local instance of `access`, identify the object `packed_query_head_row_view` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flash-attention/delta/grouped_head_row_packing@1/template.yaml`. Original outline: `knowledge/studies/flash-attention/delta/grouped_head_row_packing@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: Query heads sharing K/V can occupy one row tile and reuse its operands.. Storage: Original backing allocation through a hierarchical view; no mandatory copy.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `bijection_and_address` (proof): Packed rows and KV-head coordinates are a bijection with original token/query-head pairs, preserving every element address.
- `semantic_metadata` (test): Masks, callbacks, statistics, output and derivative state consistently decode the original token/head identity.
- `native_layout_and_gradient` (test): Backend layout/rank lowering and any K/V gradient reduction preserve the complete grouped-head contract.

The view preserves values; a changed contraction/reduction association in a consuming kernel requires separate numerical admission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
