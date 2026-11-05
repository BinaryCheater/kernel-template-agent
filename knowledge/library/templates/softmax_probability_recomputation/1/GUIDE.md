# softmax_probability_recomputation@1: row normalization checkpoint

## When to consider

When a later consumer needs softmax probabilities but retaining the full matrix is expensive, save a row log-normalizer and reconstruct score tiles on demand. Trade repeated score arithmetic for matrix storage and traffic. Match the forward mask, scale and log base, keep operands unchanged, and admit the actual checkpoint, exponent and cast errors in the numerical policy.

## Computation before and after

Recompute normalized probability tiles from compact row checkpoints.

Replace storage and later reads of the full normalized probability matrix with a saved log-normalizer per logical row. For a tile, recompute its scaled logits x from the unchanged operands. With a natural-log checkpoint L use P=exp(x-L); with base2 checkpoint L2 use P=exp2(x*log2(e)-L2). Explicitly set excluded logical positions to zero, apply admitted probability casts, and feed the reconstructed tile to its consumer before releasing it. Tile traversal must use the same row normalization domain as forward, even when the backward kernel owns a transposed key/query tile.

Recover the original computation: Materialize each reconstructed tile in its logical row/column position; under the checkpoint and numerical premises it is the probability matrix that the original consumer would read.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Deterministic softmax score reconstruction with per-row finite log-normalizers. The TileLang representative saves FP32 base2 LSE and recomputes FP16 QK into FP32 fragments. Random dropout, changed operands, unsupported empty rows and bitwise identity across arbitrary MMA/cast orders are outside this version.

Scope predicate: `p.checkpoint_matches_forward and p.operands_unchanged and p.mask_and_scale_match and p.valid_rows_have_positive_finite_mass and p.checkpoint_log_base_known and p.tail_rows_guarded and p.recomputed_logits_match and p.recompute_math_and_casts_admitted and p.checkpoint_precision_admitted`.

- Precondition `pre_checkpoint` (semantic): `p.checkpoint_matches_forward and p.operands_unchanged and p.mask_and_scale_match` — Recomputed rows use the same forward operands, score scale, logical support and saved normalization domain.
- Precondition `pre_mass` (semantic): `p.valid_rows_have_positive_finite_mass and p.checkpoint_log_base_known and p.tail_rows_guarded` — Every reconstructed valid row has a defined finite log-normalizer and padded rows cannot affect results.
- Precondition `pre_numeric` (implementation): `p.recomputed_logits_match and p.recompute_math_and_casts_admitted and p.checkpoint_precision_admitted` — The numerical policy explicitly admits score recomputation, saved-statistic precision and probability conversion.

## Local example and misuse

For a local instance of `map`, identify the object `row_normalization_checkpoint` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/tilelang/delta/softmax_probability_recomputation@1/template.yaml`. Original outline: `knowledge/studies/tilelang/delta/softmax_probability_recomputation@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One log-normalizer per row plus the currently consumed tile replaces a full probability matrix.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `normalization_identity` (proof): The checkpoint and recomputed scaled logits reconstruct the same normalized logical row.
- `checkpoint_binding` (test): Forward saved statistics and backward operand/mask/scale bindings refer to the same logical rows and normalization support.
- `numerical_reconstruction` (test): Checkpoint rounding, score replay, exponent approximation and consumer casts satisfy the declared numerical policy.

Recomputed probabilities need not be bitwise equal to probabilities rounded or generated in a different forward MMA order.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- enables: [operator_fusion@2](../../operator_fusion/2/GUIDE.md), [operator_fusion@3](../../operator_fusion/3/GUIDE.md), [operator_fusion@4](../../operator_fusion/4/GUIDE.md); Reconstructed probability tiles may feed their consumer on chip without a global probability matrix.

Structured contract: [template.yaml](template.yaml).
