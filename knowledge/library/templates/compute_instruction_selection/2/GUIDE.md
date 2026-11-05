# compute_instruction_selection@2: compute instruction tiling

## When to consider

A tiled contraction has known operand types and layout, and hardware matrix instructions can replace scalar products without violating its numerical contract.


## Computation before and after

contraction lowered to fixed-shape compute instructions

Warp (or warpgroup / thread) tile loops over instruction tiles and issues one instruction per tile; fragments use the instruction's register layout. In TileLang: T.gemm on shared-memory tiles into a register fragment.

Recover the original computation: The sum over instruction tiles of each instruction's product reproduces the executor tile's contraction.

## Bind the rule

1. Caller fixes FP16 A/B, FP32 accumulation, FP16 output and physical strides for the replay GEMM. The output cast is outside this instruction-selection rule.
2. Select a target-supported instruction family and its operand-source ABI. For mma_sync_m16n8k16_f16 bind instruction accordingly, operand_source=rr, accum_dtype=f32, macs_per_instruction=16*8*16=2048.
3. Bind p.operand_bits=p.instr_operand_bits=16 and p.operand_exp_bits=p.instr_exp_bits=5, and establish p.operand_dtype_supported from the actual instruction contract. Filter by target capability even though requires is empty.
4. Make executor extents multiples of instruction extents using zero-filled padded loads, and map each lane/register operand and accumulator coordinate. Establish exact-once coverage; inspect lowering when a compiler supplies that mapping.


| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| instruction | compile / boundary | ["mma_sync_m16n8k8_tf32", "mma_sync_m16n8k16_f16", "mma_sync_m16n8k16_bf16", "wgmma_m64nNk16_f16", "fma_f32", "tcgen05_mma"] | compute instruction family and shape |
| operand_source | compile / software | ["rr", "rs", "ss"] | where A and B operands are read from (registers or shared memory descriptors) |
| accum_dtype | compile / software | ["f32", "f16"] | accumulator precision |
| macs_per_instruction | compile / boundary | {"lo": 1, "hi": 262144} | multiply-accumulates per instruction (m*n*k of the instruction shape; 1 for scalar FMA) |

## Coupled constraints and scope

Contractions and dot products whose operand type (bit width and exponent bits, from the Problem's params) is the type the chosen instruction accepts (p.instr_*, bound by the chain); the instruction domain is further filtered by the Target's HwFeatures.

Scope predicate: `p.operand_bits == p.instr_operand_bits and p.operand_exp_bits == p.instr_exp_bits`.

- Precondition `pre_dtype` (semantic): `p.operand_dtype_supported` — operand dtypes are accepted by the instruction (after any conversion step)
- Constraint `c_macs` (semantic): `macs_per_instruction >= 1` — instruction performs at least one multiply-accumulate

## Local example and misuse

A local 32x64x32 executor contraction contains (32/16)*(64/8)*(32/16)=32 m16n8k16 instructions. Their 32*2048 products equal 32*64*32. The instruction's B fragment convention must consume the row-major caller's B through a legal layout mapping; changing the caller ABI to make a fragment convenient is invalid. The logical binding is illustrative and does not assert compilation on the replay target.


## Implementations, evidence and extension boundary

The seed mentions TileLang T.gemm, mma.sync, WGMMA and tcgen05 families, but these have different target and collective requirements. The concrete m16n8k16 example is an ISA-level binding; a TileLang implementation may choose a different supported atom. Validate compiler support and fragment layouts separately. A newly authored outline accompanies this unchanged seed contract; it contains no solver results or GPU assurance.


Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/seed/templates/compute_instruction_selection.yaml`. Original outline: `consolidation-authored`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Matrix instructions increase product throughput and operand reuse while fixing fragment shapes and collective participation. Padding, conversion, register pressure and layout exchange can outweigh the benefit on narrow or tail-heavy tiles. FP16 accumulation is excluded by the replay task even though it is in the generic domain.


```yaml
transactions:
- id: mma
  pattern: contract
  kind: instruction
  count: p.M * p.N * p.K / macs_per_instruction
  amounts:
    flops: 2 * macs_per_instruction
storage: []
overlap: []
```

## Obligations and numerical checks

- `fragment_layout` (test): Each lane holds exactly the operand and accumulator elements the instruction defines for it.
- `exact_once` (exhaustive): Every (m, n, k) product of the executor tile is issued in exactly one instruction.

Instruction accumulation order and internal precision differ from a sequential FP32 sum; covered by the contract tolerance.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- requires: [output_tiling@2](../../output_tiling/2/GUIDE.md); instruction tiles subdivide an executor tile

Structured contract: [template.yaml](template.yaml).
