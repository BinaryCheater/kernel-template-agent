# butterfly_linear_transform@1: normalized butterfly block

## When to consider

Use for a requested normalized Hadamard transform on complete power-of-two blocks. Bind H to the transform length. Choose V values per lane and a compatible thread count; H/V lanes own one block. With H=32 and V=4, eight lanes cooperate. V changes which pairwise additions and subtractions stay in registers and which require communication. Preserve the transform order and final normalization.

## Computation before and after

Register then XOR-shuffle then shared-memory sum/difference butterfly.

Treat each lane's V values as contiguous indices. Apply log2(V) register butterflies pairing index i with i xor 2**stage. Then apply log2(H/V) lane butterflies, using XOR shuffle for the first five lane stages and shared memory for later stages. Each pair produces low+high and low-high. Shared stages have both write-before-read and read-before-reuse barriers. Multiply all outputs by 1/sqrt(H).

Recover the original computation: Each stage is the corresponding factor of the Sylvester Hadamard matrix; composing the factors and normalization yields the requested block transform.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| H | compile / boundary | {"lo": 2, "hi": 512, "pow2": true} | Transform block extent fixed by the requested normalized Hadamard operation. |
| V | compile / software | {"lo": 1, "hi": 32, "pow2": true} | Values per lane; chooses the boundary between register and cross-lane butterflies. |
| threads | compile / software | {"lo": 32, "hi": 1024, "step": 32} | Participating CTA threads for one or more complete transform blocks. |

## Coupled constraints and scope

A requested normalized block Hadamard transform, not an unrestricted change to a GEMM operand. H is a boundary parameter; V and thread grouping change the implementation. The bounded candidate domain is supported by the source's power-of-two conditions; register capacity and exact compiled layouts remain target-specific obligations.

Scope predicate: `p.normalized_hadamard`.

- Precondition `pre_requested` (semantic): `p.normalized_hadamard and p.length % H == 0` — The coarse operation already requests this transform on complete blocks.
- Constraint `c_blocks` (semantic): `H % V == 0 and threads % (H // V) == 0` — Each block consists of a whole number of lanes and the CTA holds whole blocks.
- Constraint `c_lanes` (implementation): `H // V <= 1024 and (H // V <= 32 or (H // V) % 32 == 0)` — Cross-warp blocks contain whole warps.

## Local example and misuse

For a local instance of `map`, identify the object `normalized_butterfly_block` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/humming/delta/butterfly_linear_transform@1/template.yaml`. Original outline: `knowledge/studies/humming/delta/butterfly_linear_transform@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Registers and shared scratch for blocks wider than one warp.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `matrix_identity` (exhaustive): Every output coefficient equals the normalized Walsh sign for its logical input and output indices.
- `no_cross_block` (exhaustive): Pairing never mixes independent transform blocks even when several occupy one CTA.
- `scratch_lifetime` (test): Shared-stage writes complete before reads and no next-stage write destroys a live value.
- `numeric_error` (test): FP32 additions and rsqrt normalization satisfy the caller's transform tolerance.

Reassociation and approximate inverse square root alter FP32 rounding; finite exact basis checks cover indices only.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
