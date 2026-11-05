# transpose_attention_chain@1: token major transposed attention chain

## When to consider

When decoding one token leaves the MMA query dimension too small, place KV tokens on the score MMA's large first dimension and query heads sharing that KV head on its small second dimension. Compute K*Q^T, normalize each head column over tokens, then V^T*P^T and transpose the output indices back. Preserve mask/callback and LSE axes. This is a contraction-chain rewrite, not merely a rank-three strided view; reducing over heads after the swap is invalid.

## Computation before and after

Transpose both attention contractions and preserve the token-normalization axis.

For a query-head group Q[H,D], K[T,D], V[T,Dv], replace score S[H,T]=Q*K^T with St[T,H]=K*Q^T. For every head h, apply its original scale, score modifier and logical mask at St[t,h], then normalize down the token axis t to form Pt[T,H]. Compute Ot[Dv,H]=V^T*Pt and store O[h,dv]=Ot[dv,h]. LSE remains one value per logical head, never one value per token. Tile the token and value axes to use the backend's large first MMA dimension, with head padding and subgroup output/metadata guards. This rewrites the contraction chain; any physical transpose/copy and register/TMEM layout are separate bindings.

Recover the original computation: Transpose St,Pt and Ot back. Column-normalization of St equals row-normalization of S, and (P*V)^T=V^T*P^T, recovering the original grouped attention chain.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Single-position grouped-query decode with a shared KV head per query-head group. Source SM100 uses head tile1/2/4/8/16/32, token tile128/256 and 16-aligned head/value dimensions. Other token/head counts need their own implementation binding. This does not authorize row-wise softmax after the transpose, mixing distinct KV heads, arbitrary prefill reshape or a zero-cost physical transpose assumption.

Scope predicate: `p.single_query_position_per_head and p.query_head_group_shares_kv and p.softmax_reduces_tokens_per_query_head and p.mask_and_score_mod_coordinates_remapped and p.metadata_axes_remapped and p.transposed_operand_layouts_supported and p.padded_query_heads_excluded and p.empty_support_contract_preserved and p.qk_and_pv_order_casts_and_scaling_admitted`.

- Precondition `pre_group` (semantic): `p.single_query_position_per_head and p.query_head_group_shares_kv` — Each query head contributes one query vector, and heads grouped into the small score dimension consume the same K/V sequence.
- Precondition `pre_axes` (semantic): `p.softmax_reduces_tokens_per_query_head and p.mask_and_score_mod_coordinates_remapped and p.metadata_axes_remapped` — Normalization still runs over tokens independently for each query head; masks, score callbacks, LSE and output indices preserve their logical coordinates.
- Precondition `pre_backend` (implementation): `p.transposed_operand_layouts_supported and p.padded_query_heads_excluded and p.empty_support_contract_preserved` — The backend supports the new K/Q/V/P orientation and excludes padded head lanes without changing empty-row behavior.
- Precondition `pre_numeric` (implementation): `p.qk_and_pv_order_casts_and_scaling_admitted` — Reoriented contractions, probability casts, descales and their reduction order satisfy the numerical contract.

## Local example and misuse

For a local instance of `contract`, identify the object `token_major_transposed_attention_chain` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flash-attention/delta/transpose_attention_chain@1/template.yaml`. Original outline: `knowledge/studies/flash-attention/delta/transpose_attention_chain@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: Query heads sharing one KV sequence become the small MMA output dimension.. Storage: Backend-dependent transposed views, staged P and output accumulators.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `contraction_identity` (proof): Every head/token score and head/value output has the same contraction operands and admitted numerical order after the transpose rewrite.
- `normalization_and_metadata` (test): Softmax reduces the token axis for each head and all modifiers, masks, LSE and output stores use the original logical coordinates.
- `backend_layout` (test): Physical operand layouts, padded heads and fragment redistributions implement the transposed chain without alias/lifetime violations.

Equivalent real contractions can use different MMA association and probability conversion order; normalization over the wrong axis changes the operator completely.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
