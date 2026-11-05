# multi_buffer_rotation@2: buffer slot

## When to consider

Successive loop generations reuse a buffer, and prefetching or delayed refills must avoid overwriting values still read by another participant.


## Computation before and after

rotate generations through S slots, loading P ahead

for u in 0..P-1: load generation u -> slot u mod S; for p in 0..n-1: if p+P < n and not late: load generation p+P -> slot (p+P) mod S; barrier (leading); consume slot p mod S; if beta: barrier (trailing); if p+P < n and late: load generation p+P -> slot (p+P) mod S. With private every thread loads and consumes its own slots.

Recover the original computation: Forget the slots and the schedule: each consumption of slot p mod S reads generation p, so the computation equals the single-buffer one.

## Bind the rule

1. Caller supplies p.iterations and actual shared/private ownership. Choose S slots and P generations of lookahead; beta selects a trailing CTA barrier, late changes refill placement, private describes thread-private storage.
2. Check P <= S-2+beta+private-beta*private+late and late<=P. These formulas assume this rule's leading barrier and event schedule; they are not universal asynchronous-pipeline formulas.
3. Generation g lives at g%S. Define prologue loads, every readiness wait, last read and refill, plus a drain with no out-of-range future generation.
4. Apply a separate legal transfer implementation when P>=1 should overlap. Do not set private=1 merely to pass an inequality for shared storage.


| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| S | compile / software | {"lo": 1, "hi": 8} | number of slots |
| P | compile / software | {"lo": 0, "hi": 7} | prefetch distance in generations |
| beta | compile / software | [0, 1] | 1 when a trailing barrier follows each consumption before the slot is refilled |
| late | compile / software | [0, 1] | 1 when the refill of generation p+P is issued after consuming p (after the trailing barrier when beta), not before the leading barrier |
| private | compile / software | [0, 1] | 1 when every thread owns its slots (register fragments): only its own program order protects them, no other consumer |

## Coupled constraints and scope

A value consumed in generations by a loop, with a leading barrier before each consumption and loads that may be issued ahead (needs async_transfer for P >= 1 to overlap). Barrier semantics are those of a CTA-wide barrier. Thread-private register buffers (private = 1) and refills issued after the consumption (late = 1) are covered since version 2 (SYSDEBUG-2 D704; the v1 obligation rejected the correct register double buffering of CUTLASS and Marlin and the single-slot refill of FlashAttention-2).

Scope predicate: `p.iterations >= 1`.

- Precondition `pre_lifetime` (semantic): `P <= S - 2 + beta + private - beta * private + late` — generation correctness; with pre_late the weakest precondition, exact on the event model over the declared domains (the paper's 24 cases at late = private = 0)
- Precondition `pre_late` (semantic): `late <= P` — readiness of a late refill
- Constraint `c_lifetime` (semantic): `P <= S - 2 + beta + private - beta * private + late` — a slot is refilled only after every consumer finished the generation it held; P <= S - 2 + max(beta, private) + late (paper Listing C3 when late = private = 0)
- Constraint `c_late` (semantic): `late <= P` — a refill issued after the consumption cannot be the generation just consumed
- Constraint `c_capacity` (resource): `S * p.size_bytes <= p.capacity_bytes` — S generations fit the capacity budget of the level (shared memory per block for smem tiles)

## Local example and misuse

S=2,P=1,beta=1,late=0,private=0 satisfies 1<=1. For three iterations, prologue loads generation0; the main loop can issue generation1 into slot1, wait before reading generation0 in slot0, then use the trailing barrier before any destructive refill of slot0. Generation2 reuses slot0. Omitting beta without another contract gives 1<=0, so this binding rejects the schedule. With a long K tail, slot generation and valid-element count are separate facts.


## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/seed/templates/multi_buffer_rotation.yaml`. Original outline: `knowledge/seed/proofs/multi_buffer_rotation.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Prefetching may hide latency but S multiplies staged storage; beta adds synchronization. A compiler's pipeline or mbarrier protocol is supported only after its events are matched to this schedule; do not copy the inequality onto an unrelated pipeline.


```yaml
transactions:
- id: barrier
  pattern: sync
  kind: cta_barrier
  count: p.iterations * (1 + beta)
  amounts:
    warps: p.warps
  note: leading barrier, plus the trailing one when beta
storage: []
overlap:
- between:
  - pattern:transfer
  - pattern:contract
  when: P >= 1
  inflight: P
  trips: p.iterations
```

## Obligations and numerical checks

- `generation_correctness` (exhaustive): Every consumer reads the generation it expects: slot (u mod S) holds generation u from the completion of its load until the last read of u.
- `progress` (proof): The number of unconsumed generations decreases at every phase (rank rho); no wait is circular.

No numeric risk is declared by this proposal.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- requires: [physical_storage@1](../../physical_storage/1/GUIDE.md); rotates the generations of a buffer

Structured contract: [template.yaml](template.yaml).
