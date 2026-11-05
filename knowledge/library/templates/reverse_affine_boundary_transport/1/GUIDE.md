# reverse_affine_boundary_transport@1: distributed affine recurrence adjoint boundaries

## When to consider

When recurrent training spans rank fragments, summarize each local backward with zero future adjoint and use transposed forward maps to transport future contributions backward. Choose owner gather/chain/scatter or a supported short neighbor chain; preserve predecessor identity, exclusive seeds and map generations.

## Computation before and after

Resolve recurrent training exit adjoints by a reverse affine chain of transposed forward maps and zero-exit local offsets.

For each document/head, form d_f by running local reverse recurrence with zero exit adjoint. The actual entry adjoint is M_f^T*lambda_out+d_f. With head_owner_exchange, send d_f and, unless already saved at that owner, matching transfer maps. Wait before reads. Starting at the final fragment offset d_last (terminal exit adjoint zero), walk fragments in reverse chronological order, publishing the current accumulated adjoint as the exit adjoint of each predecessor before applying that predecessor map/offset. Scatter those boundaries back and wait before local seeded reverse walks. A two-fragment document needs only the last offset, not its unused transfer. With neighbor_hops, the last fragment sends d_last to its document predecessor; at each later hop the receiving fragment applies its own M_f^T*lambda_out+d_f and sends to its predecessor. Enforce the supported hop bound and local map availability. Fragments at document ends use zero exit seeds and can execute their tape walks while transport runs, exactly once. Interior local subsegment adjoints are reconstructed in reverse order from their boundary, with the same map orientation. Preserve document/head identity and matching generation through every pool offset; holes and padded heads are not consumed.

Recover the original computation: Substitute each fragment affine adjoint map into the reverse document chain. The returned boundary for fragment f is exactly the adjoint produced by all fragments strictly after f; applying its own local seeded reverse walk then recovers its full contribution. Both transport organizations preserve this noncommutative order.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| boundary_transport | compile / software | ["head_owner_exchange", "neighbor_hops"] | Gather summaries to a head owner for reverse ordered chains and scatter boundaries, or propagate completed adjoints along document predecessor hops. |
| transfer_source | compile / software | ["saved_forward_map", "resent_local_map"] | Owners reuse matching forward transfer tiles, or local fragments reconstruct and send maps with their backward offsets. Neighbor hops require the local-map form. |

## Coupled constraints and scope

FlashInfer deterministic KCP GDN SM100/103/107, 128-key/value state and grouped Q/K heads, with zero document terminal adjoint. Head-owner all-to-all supports saved or reconstructed maps; source ring requires n_d_max-1<=2 and save_transfer=False. This rule describes distributed boundary propagation, not arbitrary network collectives or a hardware performance claim. Exact tests establish affine/order identities; native transport progress and numerical equality remain implementation obligations.

Scope predicate: `(p.max_document_rank_hops >= 0) and (p.fragment_forward_map_and_zero_exit_adjoint_offset_explicit) and (p.document_fragment_order_head_ownership_and_predecessor_maps_complete) and (p.transfer_orientation_and_generation_match_backward_fragment) and (p.summary_and_transposed_chain_numeric_contract_admitted) and (boundary_transport != 'head_owner_exchange' or p.owner_collective_splits_order_and_progress_valid) and (boundary_transport != 'neighbor_hops' or (p.max_document_rank_hops <= 2 and transfer_source == 'resent_local_map' and p.document_neighbor_hop_tables_and_p2p_progress_valid)) and (p.summary_reads_and_exit_adjoint_consumers_ordered_before_storage_reuse) and (p.selected_target_partition_and_transport_paths_implemented)`.

- Precondition `pre_affine` (semantic): `p.fragment_forward_map_and_zero_exit_adjoint_offset_explicit` — Each fragment has S_out=M_f*S_in+B_f in column-state convention. Its local backward with zero exit adjoint gives d_f, so its entry adjoint is M_f^T*lambda_out+d_f. The source document terminal adjoint is zero.
- Precondition `pre_order` (implementation): `p.document_fragment_order_head_ownership_and_predecessor_maps_complete` — Chronological fragment chains and their reverse predecessor maps are complete, disjoint per document/head and independent of rank numbering. Owner capacity padding is never a live head.
- Precondition `pre_maps` (implementation): `p.transfer_orientation_and_generation_match_backward_fragment` — Saved forward M_f^T or resent backward M_f storage must resolve the same mathematical M_f^T operator, with explicit load orientation, matching local inputs and numeric generation.
- Precondition `pre_numeric` (implementation): `p.summary_and_transposed_chain_numeric_contract_admitted` — Zero-seed local reverse summaries and native FP32/TF32 stages implement the declared adjoint approximation. Exact affine algebra does not establish bitwise agreement with a reassociated sequential backward.
- Precondition `pre_owner` (implementation): `boundary_transport != 'head_owner_exchange' or p.owner_collective_splits_order_and_progress_valid` — Owner mode sends local adjoint summaries to head owners, waits before merging, then returns the exclusive exit adjoint to each preceding fragment. All ranks including empty head owners use matching collective order and compact split sizes.
- Precondition `pre_ring` (implementation): `boundary_transport != 'neighbor_hops' or (p.max_document_rank_hops <= 2 and transfer_source == 'resent_local_map' and p.document_neighbor_hop_tables_and_p2p_progress_valid)` — Source neighbor-hop mode is restricted to documents spanning at most three ranks, with local maps available for each hop and save_transfer disabled. Hop count follows document neighbors, not a cyclic rank permutation.
- Precondition `pre_lifetime` (implementation): `p.summary_reads_and_exit_adjoint_consumers_ordered_before_storage_reuse` — Wait before consuming received data, publish each boundary before local tape walks and retire transport/map storage after its last reader. Overlapped document-end walks run exactly once; their aliased value-adjoint updates are not idempotent.
- Precondition `pre_scope` (implementation): `p.selected_target_partition_and_transport_paths_implemented` — Selected backend implements the partition, local segment carries, head groups, exchange and summary layout. No crossing fragments requires no boundary exchange; it still performs local backward.
- Constraint `c_hops` (implementation): `p.max_document_rank_hops >= 0` — Maximum predecessor links in a document chain is nonnegative.

## Local example and misuse

For a local instance of `scan`, identify the object `distributed_affine_recurrence_adjoint_boundaries` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/reverse_affine_boundary_transport@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/reverse_affine_boundary_transport@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Owner summary/return pools or hop staging, with saved transposed forward maps or resent local maps.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `transpose_chain` (proof): Reverse summary composition is the transpose pullback of chronological forward maps, with local d_f injection and exclusive future boundary seeds.
- `boundary_placement` (test): Every returned boundary reaches the correct predecessor/document/head and no padded owner slot is consumed.
- `transport_progress` (test): Owner collectives or bounded document-neighbor hops have matching sends/receives, supported progress and waits before reads.
- `numeric_and_lifetime` (test): Map orientation, FP32/TF32 stages, generation and once-only aliased tape walks satisfy the native numeric and last-reader contract.

A missing transpose, chronological rather than reverse map order, stale transfer or wrong predecessor produces incorrect gradients. FP32 affine chains are not automatically bitwise equivalent to sequential local backward.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
