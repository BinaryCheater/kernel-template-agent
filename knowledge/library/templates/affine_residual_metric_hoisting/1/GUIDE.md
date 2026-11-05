# affine_residual_metric_hoisting@1: affine residual metric hoisting

## When to consider

When each scanned vector decodes as centroid plus feature offset plus scaled code, hoist query-invariant and probe-invariant terms while retaining metric-specific normalization. For query5,centroid2,vmin1,code3,scale0.5, decoded4.5 gives L2=0.25 through (5-1-2-1.5)^2. Using query-centroid without vmin gives a different result; cosine must also include4.5 squared in its denominator.

## Computation before and after

When each scanned vector decodes as centroid plus feature offset plus scaled code, hoist query-invariant and probe-invariant terms while retaining metric-specific normalization.

Cache scale once per query. For L2, cache query-vmin and per probe subtract centroid to obtain query_term; sum (query_term-code*scale)^2. For IP/cosine, cache query and per probe cache centroid+vmin; sum query*(aux+code*scale), and for cosine additionally sum decoded components squared. Finalize L2 directly, IP by negation for select-min then undo sign on return, cosine as 1-dot/(query_norm*sqrt(decoded_norm_sq)) with requested zero-denominator result. Shared storage can be reused for top-k merge only after the final distance read barrier.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| vector_bytes | compile / software | [16] | Observed per-lane SQ code-load chunk; padded dimension rounds to this width. |

## Coupled constraints and scope

NVIDIA IVF-SQ8 residual scan. Unlike Faiss midpoint SQ, this decoder has no implicit half-step; format conversion and metric normalization must not be conflated. Source cosine zero denominator returns zero. Resource effects: Three dim-length shared float arrays per CTA, vectorized code loads and local top-k or full materialized output. Hoisting amortizes query/probe work across all vectors in a list.

Scope predicate: `p.affine_decoder_contract and p.metadata_feature_aligned and p.hoisted_buffers_distinct and p.phase_barriers and p.metric_policy_and_rounding_admitted`.

- Precondition `pre_0` (semantic): `p.affine_decoder_contract and p.metadata_feature_aligned` — Decoded component is centroid[d]+vmin[d]+code[d]*scale[d] with matched metadata.
- Precondition `pre_1` (semantic): `p.hoisted_buffers_distinct and p.phase_barriers` — Query-invariant and per-probe shared arrays are distinct and synchronized before reads/reuse.
- Precondition `pre_2` (semantic): `p.metric_policy_and_rounding_admitted` — L2, IP or cosine convention including zero denominator and negated ranking is explicitly requested.

## Local example and misuse

For a local instance of `map`, identify the object `affine_residual_metric_hoisting` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cuvs/delta/affine_residual_metric_hoisting@1/template.yaml`. Original outline: `knowledge/studies/cuvs/delta/affine_residual_metric_hoisting@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Three dim-length shared float arrays per CTA, vectorized code loads and local top-k or full materialized output. Hoisting amortizes query/probe work across all vectors in a list.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `algebra` (proof): Hoisted expression equals requested affine-decoded metric before admitted rounding.
- `lifetime` (proof): Probe overwrite follows all previous readers; merge scratch aliases shared metadata only after last use.
- `normalization` (proof): Cosine uses decoded-vector norm, not original/prequantized vector norm.
- `tails` (proof): Padded code loads stay inside allocated padded storage and dimensions beyond the logical size do not accumulate.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
