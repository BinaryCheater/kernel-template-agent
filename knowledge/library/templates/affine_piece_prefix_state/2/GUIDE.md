# affine_piece_prefix_state@2: affine piece summary

## When to consider

When independent recurrence pieces have right-affine summaries, choose how to supply their entry states. Each piece may replay its preceding summary prefix, or one shared ordered fixup can store all boundaries before parallel output replay. Preserve noncommutative order and use exclusive entry states despite inclusive boundary storage.

## Computation before and after

Compose right-affine piece summaries and supply exclusive entries by per-piece prefix replay or one shared ordered fixup.

Partition ordered chunks into contiguous pieces. Each piece starts summary L=I,c=0 and folds its chunk mapsS_next=S*Lj+cj in order: L=L*Lj, c=c*Lj+cj. Publish complete matching-generation summaries. With prefix_supply=piece_replay, piece r reconstructs its entry by starting S0 and applying preceding piece maps0..r-1 in order, then executes its own local chunk recurrence/output. With shared_fixup, a separate ordered pass starts S0 and applies each piece map once, storing inclusive boundary F_r after piece r. Publish all boundaries before parallel local execution. Piece0 uses S0, piece r>0 uses F_(r-1), never its own inclusive F_r. Fixed-state capacity holes are not boundary records. Only the last valid piece writes final state; each local output has one owner. Summary and fixed-boundary precision/casts follow the declared numerical recipe. Short final pieces include only valid chunks; identity padding does not advance history.

Recover the original computation: Expand each ordered summary into constituent chunk maps. Both prefix supplies give piece r exactly maps0..r-1 applied to its matching S0; shared storage F_r includes piece r and is therefore consumed by r+1. Local execution then recovers the original recurrence under the admitted summary/fixup rounding policy.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| piece_chunks | runtime / software | {"lo": 1, "hi": 65536} | Consecutive inner recurrence chunks per piece, runtime in FlashInfer CP; constant specialization recovers the extent choice of version1. A short final piece retains only its valid chunks. |
| prefix_supply | compile / software | ["piece_replay", "shared_fixup"] | Each piece reconstructs its own prefix from preceding summaries, or one ordered fixup produces every inclusive boundary state for later parallel piece execution. |

## Coupled constraints and scope

Ordered right-affine matrix-state recurrences. Version1 piece-replay remains addressable: SGLang KDA64-token chunks, BF16 pieceL/FP32 piecec and TF32 prefix application on SM100a/103a. Shared-fixup witness: FlashInfer CP GDN SM90a,64-token inner blocks, runtime64-aligned piece lengths, FP32 transfer/offset/fixed-state records, FP32 SIMT or TF32 HMMA fixup, followed by parallel local output. Summary construction and fixup precision must be admitted. This is not a commutative reduction or a distributed collective; bitwise equivalence between supplies/backends is outside the rule.

Scope predicate: `(p.chunk_state_update_is_right_affine and p.complete_ordered_chunk_partition and p.summary_storage_ready_before_use) and (p.summary_precision_and_reassociation_admitted and p.producer_consumer_progress_supported) and (prefix_supply == 'piece_replay' or p.shared_fixup_inclusive_boundaries_and_exclusive_piece_entry_maps_agree) and (p.summary_generation_coherent_and_all_boundary_writes_publish_before_local_replay)`.

- Precondition `pre_affine` (semantic): `p.chunk_state_update_is_right_affine and p.complete_ordered_chunk_partition and p.summary_storage_ready_before_use` — Chunk j has state updateS_next=S*L_j+c_j; local outputs may depend on the entry state, and summaries use the same ordered factors.
- Precondition `pre_numeric` (implementation): `p.summary_precision_and_reassociation_admitted and p.producer_consumer_progress_supported` — Summary representation and matrix arithmetic have an explicit error contract; waiting consumers cannot prevent producers from running.
- Precondition `pre_prefix_supply` (implementation): `prefix_supply == 'piece_replay' or p.shared_fixup_inclusive_boundaries_and_exclusive_piece_entry_maps_agree` — Shared fixup stores after-piece states; each local consumer selects its preceding valid piece within the same sequence, with piece0 using original seed. Storage holes and prior-sequence records are never consumed as prefix states.
- Precondition `pre_boundary_lifetime` (implementation): `p.summary_generation_coherent_and_all_boundary_writes_publish_before_local_replay` — Summary and fixed-boundary inputs retain one generation. All required producers complete before local replay and live through every consumer. Separate ordered launches may provide publication; no generic cross-CTA progress guarantee is inferred.

## Local example and misuse

For a local instance of `scan`, identify the object `affine_piece_summary` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/affine_piece_prefix_state@2/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/affine_piece_prefix_state@2/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One matrix linear map plus offset per piece; source maps and factors stay live through prefix consumers.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: fixed_prefix_boundaries
  level: dram
  scope: kernel
  bytes: p.piece_head_capacity * p.state_rows * p.state_columns * p.fixed_state_element_bytes
  when: prefix_supply == 'shared_fixup'
  note: Inclusive after-piece boundary state records. Summary L/c storage is separate;
    capacity can include unread ragged gaps.
overlap: []
```

## Obligations and numerical checks

- `compose` (exhaustive): Piece composition preserves chronological order and each chunk occurs once.
- `entry_state` (test): Each piece starts from the prefix excluding its own map, applied to the correct initial state and sequence.
- `publication_progress` (test): Summary publication precedes reads, all producers can run, and storage stays live until every prefix consumer finishes.
- `numerical` (test): Summary casts, TF32 products and reassociation satisfy the admitted source approximation.
- `exclusive_boundary_map` (test): Shared fixup computes each inclusive state once, publishes it before replay, and each piece loads only its own exclusive entry boundary.

Matrix composition and summary casts change rounding and can amplify error; source SEQ0 factors also intentionally differ, so its policy is not a byte-equivalent realization of the fused route.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
