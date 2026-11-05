# output_tiling@2: output tile

## When to consider

Independent outputs share inputs, and one executor can reuse a rectangular neighborhood; choose output ownership and tile dimensions before allocating its working set.


## Computation before and after

partition the output index space into rectangular tiles

Executor index -> tile origin; each executor loops only over the inputs its tile needs. In CUTLASS the CTA tile is ThreadblockShape and the warp tile WarpShape; in TileLang it is the T.Kernel grid with block_M x block_N. The grid is the number of tiles.

Recover the original computation: The union of all tiles' outputs, placed at their tile origins, is the output of the coarse map; the executor index is erased.

## Bind the rule

1. Caller fixes M, N, batch/head indices, output strides and dtypes. The optimizer chooses tile_m, tile_n and executor; index_map must preserve the caller's indexing.
2. Derive grid_m=ceil(M/tile_m), grid_n=ceil(N/tile_n). Map executor (bi,bj) to origin (bi*tile_m,bj*tile_n); batch/head coordinates are additional fixed axes.
3. Constrain the tile against instruction shape, accumulator storage and staged input bytes. These are composition constraints, not permission to change M or N.
4. Mask output coordinates i>=M or j>=N and all corresponding loads. Distinguish output ownership from the separate reduction partition.


| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| tile_m | compile / software | {"lo": 1, "hi": 512} | tile extent along the first output dimension (rows) |
| tile_n | compile / software | {"lo": 1, "hi": 512} | tile extent along the second output dimension (columns); 1 for row-wise operators |
| executor | compile / software | ["cta", "warpgroup", "warp", "thread"] | executor level that owns one tile |
| index_map | compile / software | ["identity", "batched", "grouped_heads", "custom"] | how the executor index maps to output coordinates (extra batch / head dimensions) |

## Coupled constraints and scope

Any operator whose outputs are independent given the inputs (GEMM outputs, attention rows, normalization rows).

Scope predicate: `True`.

- Precondition `pre_indep` (semantic): `p.independent_outputs` — output points of the refined map do not depend on each other
- Constraint `c_pos` (semantic): `tile_m >= 1 and tile_n >= 1` — non-empty tile

## Local example and misuse

Local GEMM output map: M=127, N=1000, tile_m=32, tile_n=64, executor=cta, index_map=identity gives grid 4 by 16. CTA (3,15) owns rows 96..126 and columns 960..999; padded rows 127 and columns 1000..1023 are never stored. This binding only partitions outputs; it does not choose K blocking or an instruction. A rectangular logical tile does not imply row-major shared-memory fragments.


## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/seed/templates/output_tiling.yaml`. Original outline: `knowledge/seed/proofs/output_tiling.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Larger tiles reuse more input elements per CTA but increase accumulator and shared-memory demand. Small M can leave padded rows idle and reduce available CTAs. Check actual occupancy and the number of useful tiles; this rule alone supplies no latency prediction.


```yaml
transactions:
- id: ctas
  pattern: launch
  kind: cta_launch
  count: ceil_div(p.M, tile_m) * ceil_div(p.N, tile_n)
- id: kernel
  pattern: launch
  kind: kernel_launch
  count: '1'
- id: compulsory
  pattern: transfer
  kind: dram_stream
  count: '1'
  amounts:
    bytes: p.compulsory_bytes
  src: dram
  dst: l2
storage: []
overlap: []
```

## Obligations and numerical checks

- `coverage_exact_once` (exhaustive): Every output point lies in exactly one tile; points of residue tiles beyond the problem bounds are not written.
- `index_map_bijective` (exhaustive): The executor index -> tile origin map is injective on valid tiles and hits every tile.

No numeric risk is declared by this proposal.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- enables: [output_tiling@2](../../output_tiling/2/GUIDE.md); applied again one executor level down (CTA tile -> warp tile)

Structured contract: [template.yaml](template.yaml).
