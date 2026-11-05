# semantic_block_page_fragments@1: semantic sparse blocks mapped to independent storage fragments

## When to consider

When sparse selection blocks and cache pages have different extents, split at their greatest common aligned fragment and translate each fragment independently. This permits crossing page boundaries without repacking KV. Keep semantic support separate from storage locators, and preserve invalid sentinels through the consumer decode.

## Computation before and after

Refine selected logical blocks into independently translated common-alignment fragments.

Let semantic block extent B, storage page extent P and derived F=gcd(B,P). A selected logical block b has B/F fragments. For each f in[0,B/F), compute logical token t=b*B+f*F, logical page lp=t//P and subpage u=(t%P)//F. Resolve physical page h=table[request,lp] independently for each fragment. Encode locator=h*(P/F)+u. Decode by h=locator//(P/F), token_offset=(locator%(P/F))*F, then load exactly F token rows with the declared head/feature strides. Use the same formula precomputed or on demand. Preserve fragment membership/original token positions and mask the partial logical tail. Invalid entries use a reserved negative sentinel with a separately proved out-of-bounds decode; do not assume signed division alone implements that sentinel. No physical adjacency follows from neighboring logical pages.

Recover the original computation: For every valid fragment element k<F, decoded physical token h*P+u*F+k equals table[request,(b*B+f*F)//P]*P+(b*B+f*F)%P+k. Concatenating its B/F fragments recovers the selected semantic block through the original page table.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| locator_mode | compile / software | ["precomputed", "on_demand"] | Materialize fragment locators ahead of the attention consumer or resolve the same mapping while supplying its index view. |

## Coupled constraints and scope

Source QToken-KvBlock-Sparse attention, semantic block4/8/16/32/64/128, storage page a positive multiple4, F=gcd(B,P). F can be smaller than either extent. Q1 has direct-view or materialized mapping; grouped queries use union/membership preparation upstream. Source valid selected prefix is distinct/in-range, not arbitrary-hole occurrence support. Other extents require a supported consumer; encoded-subpage FP8 routes may require separate split reduction rather than direct/inline/cluster publication.

Scope predicate: `(p.semantic_block_size >= 1 and p.storage_page_size >= 1 and p.fragment_size >= 1 and p.semantic_block_size % p.fragment_size == 0 and p.storage_page_size % p.fragment_size == 0) and (p.fragment_size_is_gcd_of_semantic_block_and_storage_page and p.all_blocks_pages_aligned_to_common_token_origin) and (p.selected_semantic_blocks_and_original_fragment_membership_contract_explicit) and (p.request_head_logical_block_and_page_table_generation_match and p.live_physical_pages_in_bounds) and (p.maximum_live_locator_fits_nonnegative_signed_int32_and_address_products_widened) and (p.partial_fragment_score_masks_exact_and_loaded_value_padding_finite_zero_or_other_admitted_neutral) and (p.invalid_locator_decodes_to_out_of_bounds_zero_fill_and_never_a_live_page) and ((locator_mode != 'precomputed' or p.locators_memberships_published_and_alive_through_all_consumers) and (locator_mode != 'on_demand' or p.live_mapping_inputs_immutable_until_all_resolutions_finish)) and (p.consumer_decodes_same_page_subpage_units_and_selected_mode_supported)`.

- Precondition `pre_fragment` (semantic): `p.fragment_size_is_gcd_of_semantic_block_and_storage_page and p.all_blocks_pages_aligned_to_common_token_origin` — F is the derived greatest common divisor; block and page origins share the same token coordinate frame.
- Precondition `pre_support` (semantic): `p.selected_semantic_blocks_and_original_fragment_membership_contract_explicit` — Original support, current-block inclusion and per-query membership are determined before storage translation; mapping does not deduplicate or invent support.
- Precondition `pre_map` (semantic): `p.request_head_logical_block_and_page_table_generation_match and p.live_physical_pages_in_bounds` — Every fragment uses the intended request/head and current physical page table.
- Precondition `pre_width` (implementation): `p.maximum_live_locator_fits_nonnegative_signed_int32_and_address_products_widened` — Valid locators fit nonnegative int32 and address products cannot wrap; negative sentinel stays reserved.
- Precondition `pre_padding` (implementation): `p.partial_fragment_score_masks_exact_and_loaded_value_padding_finite_zero_or_other_admitted_neutral` — Whole-fragment loading may read logical padding; later score masking alone cannot sanitize NaN/Inf V padding.
- Precondition `pre_sentinel` (implementation): `p.invalid_locator_decodes_to_out_of_bounds_zero_fill_and_never_a_live_page` — Invalid/tail slots cannot alias valid cache rows after decoding, head offset addition or integer division.
- Precondition `pre_lifetime` (implementation): `(locator_mode != 'precomputed' or p.locators_memberships_published_and_alive_through_all_consumers) and (locator_mode != 'on_demand' or p.live_mapping_inputs_immutable_until_all_resolutions_finish)` — Precomputed and direct-view paths preserve mapping generation and publication/lifetime.
- Precondition `pre_consumer` (implementation): `p.consumer_decodes_same_page_subpage_units_and_selected_mode_supported` — The consumer treats F as transfer extent and decodes P/F subpages per physical page; supported modes may have distinct target restrictions.
- Constraint `c_shape` (semantic): `p.semantic_block_size >= 1 and p.storage_page_size >= 1 and p.fragment_size >= 1 and p.semantic_block_size % p.fragment_size == 0 and p.storage_page_size % p.fragment_size == 0` — Complete physical fragments divide both semantic blocks and storage pages.

## Local example and misuse

For a local instance of `transfer`, identify the object `semantic_sparse_blocks_mapped_to_independent_storage_fragments` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/semantic_block_page_fragments@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/semantic_block_page_fragments@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Optional int32 physical-page/subpage locators and matching membership metadata.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `fragment_identity` (test): Every semantic block token maps to the correct physical page/offset even when the block crosses multiple nonadjacent pages.
- `locator_roundtrip` (test): Valid locators roundtrip without overflow, and invalid sentinels stay invalid through decode/head transforms.
- `support_and_tail` (test): Membership/current-block policy and token masks survive mapping; whole-fragment V loads have admitted padding.
- `path_equivalence` (test): Precomputed and direct-view locators use the same immutable inputs and publication generation.

Storage translation is value neutral only with exact support/masks and safe padding. A masked score multiplied by NaN V does not become a harmless zero contribution.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
