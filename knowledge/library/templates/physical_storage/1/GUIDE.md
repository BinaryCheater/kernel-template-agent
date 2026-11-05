# physical_storage@1: buffer occupancy

## When to consider

A logical intermediate needs a physical home or two intermediates may share storage because their live intervals do not overlap.


## Computation before and after

give a logical value a physical buffer at one storage level

Allocate the buffer at the chosen level (register fragment, shared-memory array, tensor memory, or global scratch); write the value into it; read it from there until its last use. Values that share a buffer are written in share_order.

Recover the original computation: Read every value from its buffer occupancy: while the occupancy lasts the buffer holds exactly the logical value, so forgetting buffers and order gives the logical dataflow.

## Bind the rule

1. Derive size, alignment and each value's producer, last read and next destructive write from the actual schedule.
2. Select level only where the target and consumers support it. Choose dedicated storage first, or shared only after proving the declared lifetime inequality.
3. Bind share_order to the selected lifetime order. Convert byte capacity and alignment to the concrete layout; logical capacity does not establish a legal fragment layout.
4. Include all consumers and asynchronous reads before declaring a value dead.
5. Bind reuse to the backend loop schedule as well as source order. A CTA barrier by itself does not establish that a compiler-pipelined alias has the declared lifetime; begin with dedicated storage or a validated serial reuse schedule.


| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| level | compile / software | ["reg", "smem", "tmem", "l2", "gmem"] | storage level that holds the value |
| buffer | compile / software | ["dedicated", "shared"] | whether the value has a buffer of its own or shares one with other values (reconstruction chains may name the buffer instead) |
| share_order | compile / software | ["none", "this_first", "other_first"] | order z of the values that share one buffer (none for a dedicated buffer) |

## Coupled constraints and scope

Any logical value with a known size; the level menu is limited by what the Target has (tmem only on chips with tensor memory).

Scope predicate: `p.size_bytes >= 1`.

- Precondition `pre_lifetime` (semantic): `p.last_read + p.delta <= p.next_write` — lifetime: the next destructive write into the buffer comes after the value's last required read (the paper's order inequality with the order already chosen)
- Constraint `c_fits` (resource): `p.size_bytes * p.copies <= p.capacity_bytes` — the copies held at the same time fit the capacity budget of the chosen level

## Local example and misuse

Stage one FP16 GEMM A tile of 32x32 and B tile of 32x64 in dedicated shared memory: logical bytes are 2048 and 4096 before padding/alignment. For aliasing, a last read at event 5, next write at event 6 and delta=1 meets 5+1<=6; moving the next write to event 5 violates it. An issued asynchronous operation may extend last_read until completion.


## Implementations, evidence and extension boundary

Replay evidence TR-A2 on GB10 with TileLang 0.1.15 exposed a concrete reuse boundary: two attention operand roles assigned to one shared tile under T.Pipelined produced incorrect results; adding CTA barriers still failed. A serial key loop with barriers and the same shared operand role passed normal, magnitude, cancellation, causal-mask and negative-logit checks. The abstract contract is unchanged: map p.last_read and p.next_write to actual completion events, including compiler scheduling and the consumers' layouts. Validate a serial realization before adding compiler pipelining; keep separate buffers when the mapping is unestablished. This observation is scoped to that recorded realization and does not supply an instruction schedule or complete attention implementation.


Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/seed/templates/physical_storage.yaml`. Original outline: `knowledge/seed/proofs/physical_storage.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: a buffer is reused by values whose lifetimes do not overlap. Storage: chosen per value: reg, smem, tmem, l2 or gmem. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: buffer
  level: level
  scope: cta
  bytes: p.size_bytes * p.copies
overlap: []
```

## Obligations and numerical checks

- `lifetime` (test): A value survives until its last read: no other value sharing the buffer is written between its first write and its last read.
- `capacity` (exhaustive): At every time, the values resident in a level fit its capacity.

No numeric risk is declared by this proposal.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- enables: [async_transfer@1](../../async_transfer/1/GUIDE.md); a value in a buffer is filled by a transfer
- enables: [multi_buffer_rotation@2](../../multi_buffer_rotation/2/GUIDE.md); a buffer can be split into rotating slots
- enables: [shared_resource_binding@1](../../shared_resource_binding/1/GUIDE.md); a buffer in banked memory gets a bank binding

Structured contract: [template.yaml](template.yaml).
