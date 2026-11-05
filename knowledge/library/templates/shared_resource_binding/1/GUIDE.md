# shared_resource_binding@1: request binding

## When to consider

A correct shared-memory layout creates bank contention, and a bijective swizzle or padding can change the bank mapping without changing logical values.


## Computation before and after

map requests to banks so concurrent requests spread across banks

Address the tile through the bank map (e.g. column chunk XOR row) on every write and read of the buffer; the hardware arbitrates the resulting requests.

Recover the original computation: The bank map is a bijection on addresses inside the tile; reading through its inverse gives the logical layout, and arbitration order does not change the values read.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| binding | compile / software | ["identity", "xor_swizzle", "padding"] | request -> bank map; reconstruction chains may name the concrete swizzle |
| queue_depth | compile / hardware | {"lo": 1, "hi": 64} | depth of the request queue in front of the banks; a chip variable, fixed on GB10 |

## Coupled constraints and scope

Accesses to banked on-chip memory with a declared arbitration contract. On GB10 the chip variables (queue depth, bank count and width) are fixed hardware values, not decisions.

Scope predicate: `hw.smem_banks >= 1`.

- Precondition `pre_progress` (semantic): `queue_depth >= 1` — a request can always enter the queue eventually (no zero-capacity queue)
- Constraint `c_queue_fixed` (resource): `queue_depth == hw.smem_queue_depth` — on a fixed chip (GB10) the queue depth is the hardware's value
- Constraint `c_min_wavefronts` (performance_assumption): `p.wavefronts == ceil_div(p.width_bytes * p.requesters, hw.smem_banks * hw.smem_bank_bytes)` — the binding reaches the minimum number of bank passes (conflict free)

## Local example and misuse

For a local shared tile, keep the logical coordinate (i,j) fixed while comparing identity and xor_swizzle physical address functions. Use distinct coordinate tags to verify the inverse mapping for every consumer. queue_depth is a hardware fact on a fixed target, not a tunable capacity that can be increased to hide a conflict. Swizzling storage without changing the matrix-load descriptor can return incorrect operands even though the bank pattern improves.


## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/seed/templates/shared_resource_binding.yaml`. Original outline: `knowledge/seed/proofs/shared_resource_binding.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: banked on-chip memory (shared memory). Structured effects describe work and resource changes, not timings.

```yaml
transactions:
- id: access
  pattern: access
  kind: smem_access
  count: p.accesses
  amounts:
    wavefronts: p.wavefronts
storage: []
overlap: []
```

## Obligations and numerical checks

- `bijective_binding` (exhaustive): The request -> bank map is a permutation of the tile's addresses.
- `conflict_freedom` (test): Each access completes in the declared number of bank passes.
- `no_deadlock_progress` (unspecified): Under the declared arbitration contract every request is eventually served.

No numeric risk is declared by this proposal.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- requires: [physical_storage@1](../../physical_storage/1/GUIDE.md); binds the requests of a buffer in banked memory

Structured contract: [template.yaml](template.yaml).
