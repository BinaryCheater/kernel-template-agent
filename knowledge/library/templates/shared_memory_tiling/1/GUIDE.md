# shared_memory_tiling@1: smem slot

## When to consider

Consider this coupled teaching contract when staging a symbolic GEMM through rotating shared-memory slots with its fixed 4-byte element and CTA-barrier model; use separate storage/instruction rules for a different numerical or hardware contract.


## Computation before and after

stage operand tiles through S shared-memory slots, prefetching P ahead

for each block (i, j): for u in 0..min(P, KT)-1: load k-tile u -> slot u mod S; for p in 0..KT-1 (KT = ceil(K/Tk)): if p+P < KT: load k-tile p+P -> slot (p+P) mod S; barrier (leading); C[i,j] += slot (p mod S); if beta: barrier (trailing). Grid ceil(M/Tm) x ceil(N/Tn); tails guarded by index < extent: loads fill 0, stores skip.

Recover the original computation: Forget the grid, the slots and the schedule: C = A B over (M, N, K).

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| Tm | compile / software | [16, 32, 64, 128, 256] | output tile rows |
| Tn | compile / software | [16, 32, 64, 128, 256] | output tile columns |
| Tk | compile / software | [8, 16, 32, 64] | reduction tile |
| S | compile / software | [1, 2, 3] | slots per block |
| P | compile / software | [0, 1, 2] | prefetch distance in k-tiles |
| beta | compile / software | [0, 1] | 1 with a trailing barrier |

## Coupled constraints and scope

C[M,N] = A[M,K] * B[K,N], C starts at 0, any M, N, K >= 1; each product is a symbolic term (the proof shows every C[i,j] gets exactly its K terms, not FP32 rounding). Chip: 96 KiB shared memory per block, 256 threads per block, 32768 accumulators per block, 4 bytes per element. Leading barrier; tiles disjoint.

Scope predicate: `p.M >= 1 and p.N >= 1 and p.K >= 1`.

- Precondition `pre_lifetime` (semantic): `P <= S - 2 + beta` — lifetime; exact on the paper's 24 checked (S, P, beta)
- Constraint `c1_capacity` (resource): `S * 4 * (Tm * Tk + Tk * Tn) <= 98304` — shared-memory capacity M_s = 96 KiB, e = 4 bytes
- Constraint `c2_accumulators` (resource): `256 <= Tm * Tn and Tm * Tn <= 32768` — accumulator budget R_acc = 32768 per block, 256 threads
- Constraint `c3_lifetime` (semantic): `P <= S - 2 + beta` — a slot is refilled only after every warp finished the tile it held

## Local example and misuse

Tm=32,Tn=64,Tk=32,S=2,P=1,beta=1 gives shared storage 2*4*(32*32+32*64)=24576 bytes and 2048 accumulators, meeting the fixed budgets and 1<=2-2+1. K=769 has 25 tiles, with only k=768 valid in the last tile. The proof counts symbolic products; this binding does not establish FP16 tensor-core arithmetic. Changing its element-size constant from 4 to 2 would change the contract and is not a consolidation edit.


## Implementations, evidence and extension boundary

This rule intentionally couples output/reduction tiles, storage, rotation and barrier placement around the paper's model. Its fixed chip constants and symbolic product arithmetic are part of its scope. The replay FP16 GEMM should prefer separate rules unless it explicitly supplies a justified implementation refinement. It is retained as an alternative, not merged with the general pipeline rules.


Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/seed/templates/shared_memory_tiling.yaml`. Original outline: `knowledge/seed/proofs/shared_memory_tiling.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: each A tile is reused Tn times and each B tile Tm times from shared memory. Storage: S slots of A and B tiles in shared memory. Structured effects describe work and resource changes, not timings.

```yaml
transactions:
- id: ctas
  pattern: launch
  kind: cta_launch
  count: ceil_div(p.M, Tm) * ceil_div(p.N, Tn)
- id: kernel
  pattern: launch
  kind: kernel_launch
  count: '1'
- id: tiles
  pattern: transfer
  kind: global_load
  count: ceil_div(p.M, Tm) * ceil_div(p.N, Tn) * ceil_div(p.K, Tk) * (Tm * Tk + Tk
    * Tn) * 4 / 16
  amounts:
    bytes: '16'
    vector_bytes: '16'
  src: global
  dst: smem
- id: fma
  pattern: contract
  kind: ffma
  count: ceil_div(p.M, Tm) * ceil_div(p.N, Tn) * ceil_div(p.K, Tk) * Tm * Tn * Tk
    / 32
  amounts:
    flops: '64'
- id: barrier
  pattern: sync
  kind: cta_barrier
  count: ceil_div(p.M, Tm) * ceil_div(p.N, Tn) * ceil_div(p.K, Tk) * (1 + beta)
  amounts:
    warps: '8'
storage:
- id: slots
  level: smem
  bytes: S * 4 * (Tm * Tk + Tk * Tn)
overlap:
- between:
  - pattern:transfer
  - pattern:contract
  when: P >= 1
```

## Obligations and numerical checks

- `coverage` (smt): Every output index and every reduction index is reached by some tile (index conditions, any extent).
- `exact_once` (smt): No index is reached by two tiles.
- `no_phantom` (smt): No admitted tile point lies beyond the extent (guarded tails).
- `readiness` (smt): Every read of slot p mod S follows the load of k-tile p, on every interleaving.
- `lifetime` (smt): Slot (u mod S) holds k-tile u from the completion of its load until its last read.
- `agreement` (smt): A complete run exists and its replay gives exactly the terms A[i,k] B[k,j] for each output.

the summation order over k differs from a sequential sum (FP32 rounding is out of the proof's scope)

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- alternative_of: [multi_buffer_rotation@2](../../multi_buffer_rotation/2/GUIDE.md); one-step form of output tiling, storage, rotation and the barrier choice (paper Listing 1)

Structured contract: [template.yaml](template.yaml).
