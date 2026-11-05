# async_transfer@1: async transfer

## When to consider

A future on-chip consumer has enough independent work to hide a global-to-on-chip transfer, and its read can be ordered after copy completion.


## Computation before and after

split a copy into issue and completion and overlap the gap with other work

issue(copy src -> dst in granularity_bytes pieces) at issue_point; ... independent work ...; wait for completion at complete_point; read dst. software: ld/st through registers; instruction: cp.async + commit/wait_group; controller: TMA copy + mbarrier wait.

Recover the original computation: Collapse issue and completion into one atomic write at the completion event; reads after the wait see exactly what the atomic write would have written.

## Bind the rule

1. Caller fixes source/destination, size and alignment. Choose software, instruction or controller only when the backend implements it; software does not imply asynchronous hardware.
2. Choose issue_point and complete_point jointly with the buffer generation schedule. Record which completion event covers which bytes and participating threads.
3. Bind granularity_bytes from a legal copy transaction and verify the declared alignment constraint plus the instruction's stricter native requirements, valid addresses and tail predicates.
4. Wait for readiness before reads, and for all consumers before buffer reuse. A copy wait and CTA publication barrier may be separate events.


| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| mechanism | compile / boundary | ["software", "instruction", "controller"] | who performs the copy: plain loads and stores, an asynchronous copy instruction (cp.async), or a controller (TMA); the paper's one-hot b_jmu |
| issue_point | compile / software | ["prologue", "ahead_by_prefetch", "interleaved_with_compute", "after_slot_released", "compiler_pipelined"] | where in the schedule the copy is issued |
| complete_point | compile / software | ["before_read", "wait_group_then_barrier", "barrier_wait", "compiler_pipelined"] | where the completion is awaited before the data is read |
| granularity_bytes | compile / software | {"lo": 1, "hi": 262144} | bytes per copy operation (16 for cp.async.cg, a whole tile for TMA) |

## Coupled constraints and scope

Transfers into on-chip storage whose destination is not read until a later point in the schedule; asynchronous copies need at least 4-byte alignment. The instruction and controller mechanisms need cp.async (sm_80+) and TMA (sm_90+) respectively.

Scope predicate: `p.align_bytes >= 4 and p.size_bytes >= 4`.

- Precondition `pre_aligned` (implementation): `p.align_bytes % granularity_bytes == 0 or granularity_bytes >= p.size_bytes` — the copy instruction's alignment requirement holds
- Constraint `c_aligned` (implementation): `p.align_bytes % granularity_bytes == 0 or granularity_bytes >= p.size_bytes` — each copy operation is aligned to its width, unless one operation moves the whole transfer
- Constraint `c_divides` (implementation): `p.size_bytes % granularity_bytes == 0` — the transfer splits into whole copy operations

## Local example and misuse

For an aligned full 32x32 FP16 A tile, size_bytes=2048, align_bytes=16, granularity_bytes=16 meets pre_aligned. An instruction copy issued ahead of use can use wait_group_then_barrier before reading, if that backend provides the promised semantics. On K=769 the last tile's invalid elements must be zero-filled with legal predicated transactions; a true tile-level alignment predicate does not make an out-of-bounds vector load legal.


## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/seed/templates/async_transfer.yaml`. Original outline: `knowledge/seed/proofs/async_transfer.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Earlier issue can hide transfer latency but increases live storage and outstanding operations. More stages consume capacity and may reduce residency. Readiness and reuse ordering are correctness constraints, not performance knobs.


```yaml
transactions:
- id: copy
  pattern: transfer
  kind: mechanism
  kind_map:
    software: global_load
    instruction: cp_async_16
    controller: tma_bulk
  count: p.size_bytes / granularity_bytes
  amounts:
    bytes: granularity_bytes
    vector_bytes: granularity_bytes
  src: global
  dst: smem
storage: []
overlap: []
```

## Obligations and numerical checks

- `readiness` (test): Every read of the destination happens after the completion of the transfer that fills it, on every interleaving.
- `no_overwrite_in_flight` (test): No transfer in flight writes a destination that is still being read (with multi_buffer_rotation this becomes P <= S - 2 + beta).
- `mechanism_available` (unspecified): The chosen mechanism exists on the Target (cp.async for instruction, TMA for controller).

No numeric risk is declared by this proposal.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- enables: [multi_buffer_rotation@2](../../multi_buffer_rotation/2/GUIDE.md); issuing ahead needs a slot to land in

Structured contract: [template.yaml](template.yaml).
