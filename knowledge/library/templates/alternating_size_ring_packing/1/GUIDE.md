# alternating_size_ring_packing@1: phase dependent three slot occupancy

## When to consider

When an odd three-slot pipeline alternates larger and smaller operand tiles, consider shifting its middle address by wrap parity instead of reserving the maximum footprint for every slot. This saves L-S bytes. Source128-row FP16 K192/V128 has L49152,S32768 and reserves131072 rather than147456 bytes. Per-slot readiness is insufficient: a newly large edge slot can overlap a still-live middle operand, so prove retirement for every overlapping range.

## Computation before and after

Pack alternating-size operands in a three-slot ring by shifting the middle slot across wrap parity.

Let L=large_bytes,S=small_bytes and logical generation g use slot i=g modulo3 and logical wrap parity h=floor(g/3) modulo2. Uniform mode uses address i*L and allocation3L. Pointer-shifted mode uses nominal stride D=(L+S)/2 and address i*D, except slot1 adds (L-S)/2 when h=0 and subtracts it when h=1. Allocation is2L+S. In parity0 the simultaneous ranges are large at0, small atL, large atL+S; parity1 uses small at0, large atS, small atL+S. Translate producer/consumer phase conventions explicitly. Before each write, retire all overlapping prior occupancies, not just the previous generation of slot i. Readiness and native async completion remain separate from address packing.

Recover the original computation: Associate each read with its logical generation and forget the phase-dependent physical address. Under readiness/lifetime premises it reads exactly the operand held by a dedicated maximum-sized slot.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| packing | compile / software | ["uniform", "pointer_shifted"] | Reserve three maximum-sized slots or pack alternating operands with a shifted middle pointer. |
| large_bytes | compile / boundary | {"lo": 1, "hi": 1048576} | Live footprint of the larger operand tile, including layout padding. |
| small_bytes | compile / boundary | {"lo": 1, "hi": 1048576} | Live footprint of the smaller operand tile, including layout padding. |

## Coupled constraints and scope

Three-slot alternating-size storage with complete footprint and lifetime bindings. Source one-CTA hd192/128 uses three KV stages and a phase-dependent middle offset; two-CTA halves payloads and disables this layout. Generic arbitrary odd-slot rings, unknown padded footprints and treating per-slot release as proof of neighboring-range retirement are outside scope.

Scope predicate: `p.operands_alternate_large_small and p.producer_consumer_logical_phase_match and p.every_overlapping_occupancy_retired_before_write and p.native_read_footprints_match_tile_footprints and p.base_and_phase_addresses_legal and p.operand_ready_before_read`.

- Precondition `pre_sequence` (semantic): `p.operands_alternate_large_small and p.producer_consumer_logical_phase_match` — Logical generation0 is large, subsequent generations alternate, and producers/consumers translate their native phases to the same logical parity.
- Precondition `pre_lifetime` (semantic): `p.every_overlapping_occupancy_retired_before_write and p.native_read_footprints_match_tile_footprints` — Before a write, all earlier async readers of any overlapping physical range have retired, including neighboring logical slots.
- Precondition `pre_backend` (implementation): `p.base_and_phase_addresses_legal and p.operand_ready_before_read` — Native transfer/MMA layouts accept the shifted addresses and consumers see complete current-generation payload.
- Constraint `c_sizes` (semantic): `large_bytes >= small_bytes and p.logical_slots == 3` — This mapping covers exactly three slots with ordered positive footprint sizes.
- Constraint `c_alignment` (implementation): `p.transfer_alignment_bytes > 0 and large_bytes % p.transfer_alignment_bytes == 0 and small_bytes % p.transfer_alignment_bytes == 0 and (packing == 'uniform' or ((large_bytes + small_bytes) % (2 * p.transfer_alignment_bytes) == 0 and (large_bytes - small_bytes) % (2 * p.transfer_alignment_bytes) == 0))` — Nominal stride and phase correction preserve the required native address alignment.
- Constraint `c_capacity` (resource): `(packing == 'uniform' and 3 * large_bytes <= p.capacity_bytes) or (packing == 'pointer_shifted' and 2 * large_bytes + small_bytes <= p.capacity_bytes)` — The selected physical allocation fits the storage budget.

## Local example and misuse

For a local instance of `store`, identify the object `phase_dependent_three_slot_occupancy` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flash-attention/delta/alternating_size_ring_packing@1/template.yaml`. Original outline: `knowledge/studies/flash-attention/delta/alternating_size_ring_packing@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Pointer-shifted mode reserves2L+S instead of3L bytes.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: uniform
  level: smem
  scope: cta
  bytes: 3 * large_bytes
  when: packing == 'uniform'
- id: shifted
  level: smem
  scope: cta
  bytes: 2 * large_bytes + small_bytes
  when: packing == 'pointer_shifted'
overlap: []
```

## Obligations and numerical checks

- `address_and_capacity` (exhaustive): All phase-specific operand footprints are aligned, in bounds and pairwise disjoint within each complete three-slot parity frame.
- `transition_lifetime` (test): Every changed physical range is free of all earlier async readers before writing, including cross-slot overlap during parity transitions.
- `phase_agreement` (test): Producer and consumer use identical logical generation, size and physical address despite any opposite native phase convention.

Address/lifetime transformation only; an admitted native layout binding preserves operand values.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- refines: [physical_storage@1](../../physical_storage/1/GUIDE.md); Supplies a concrete phase-address mapping and reduced capacity bound; the lifetime condition must include every overlapping occupancy.

Structured contract: [template.yaml](template.yaml).
