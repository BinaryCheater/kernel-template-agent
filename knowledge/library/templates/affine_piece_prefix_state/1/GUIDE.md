# affine_piece_prefix_state@1: affine piece summary

## When to consider

When chunk state updates are affine, summarize contiguous pieces and let a later piece reconstruct its entry state from earlier summaries. Choose piece size to trade summary work/storage against a shorter local state chain. Preserve noncommutative chronological order, distinguish prefix from inclusive state and admit summary precision explicitly.

## Computation before and after

Summarize contiguous affine recurrence pieces and reconstruct their prefix entry states independently.

Partition the ordered chunk sequence into contiguous pieces. In each piece initializeL=I,c=0. For each chunk map(Lj,cj), updateL=L*Lj and c=c*Lj+cj in order. Publish completed piece(L,c). To start piece r, initializeS=S0 and apply every preceding piece map in order S=S*L_piece+c_piece; then run its own local chunk recurrence and outputs from this reconstructed entry state. Consumers wait for matching piece readiness and retain summary generation identity. First piece usesS0 directly; final short pieces include only valid chunks.

Recover the original computation: Expand every piece map into its ordered constituent chunk maps. Substitution recovers the original recurrence and each local entry state in exact arithmetic; source summary/storage rounding is covered only by the admitted numerical policy.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| piece_chunks | compile / software | {"lo": 1, "hi": 65536} | Consecutive recurrence chunks per piece; source distributes a bounded piece count from shape policy, with possibly shorter final piece. |

## Coupled constraints and scope

Ordered right-affine matrix-state recurrences. Source KDA prefill uses64-token chunks,K=V128, BF16 pieceL, FP32 piecec and TF32 prefix application on SM100a/103a. No commutative scan premise; no automatic native forward-progress or exact floating-point equivalence.

Scope predicate: `p.chunk_state_update_is_right_affine and p.complete_ordered_chunk_partition and p.summary_precision_and_reassociation_admitted`.

- Precondition `pre_affine` (semantic): `p.chunk_state_update_is_right_affine and p.complete_ordered_chunk_partition and p.summary_storage_ready_before_use` — Chunk j has state updateS_next=S*L_j+c_j; local outputs may depend on the entry state, and summaries use the same ordered factors.
- Precondition `pre_numeric` (implementation): `p.summary_precision_and_reassociation_admitted and p.producer_consumer_progress_supported` — Summary representation and matrix arithmetic have an explicit error contract; waiting consumers cannot prevent producers from running.

## Local example and misuse

For a local instance of `scan`, identify the object `affine_piece_summary` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/affine_piece_prefix_state@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/affine_piece_prefix_state@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One matrix linear map plus offset per piece; source maps and factors stay live through prefix consumers.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `compose` (exhaustive): Piece composition preserves chronological order and each chunk occurs once.
- `entry_state` (test): Each piece starts from the prefix excluding its own map, applied to the correct initial state and sequence.
- `publication_progress` (test): Summary publication precedes reads, all producers can run, and storage stays live until every prefix consumer finishes.
- `numerical` (test): Summary casts, TF32 products and reassociation satisfy the admitted source approximation.

Matrix composition and summary casts change rounding and can amplify error; source SEQ0 factors also intentionally differ, so its policy is not a byte-equivalent realization of the fused route.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
