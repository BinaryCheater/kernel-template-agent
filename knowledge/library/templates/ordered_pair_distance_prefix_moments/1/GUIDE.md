# ordered_pair_distance_prefix_moments@1: ordered pair distance prefix moments

## When to consider

When weighted samples are ordered by nondecreasing scalar depth, replace quadratic pairwise absolute-distance accumulation by prefix weight and weighted-depth moments. RasterizeToPixels2DGSSerialBatchFwd.cu:411-422. Tile sort by primitive depth does not automatically prove the last color channel is nondecreasing: require caller binding.

## Computation before and after

When weighted samples are ordered by nondecreasing scalar depth, replace quadratic pairwise absolute-distance accumulation by prefix weight and weighted-depth moments.

Maintain A=sum previous weights and B=sum previous weight*depth, initially zero. For current (w,z), accumulate D+=2*w*(z*A-B), then A+=w,B+=w*z. This is sum over both orientations of all distinct sample pairs w_i*w_j*abs(z_i-z_j). For alpha compositing w=alpha*T_before and A=1-T_before only when initial T=1 and no inconsistent skipped-T updates.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Sorted scalar depths, finite admitted arithmetic, no diagonal self-interval term. Source 2DGS stores last color channel as depth and uses A=1-T. Resource effects: O(1) state and O(n) work instead of pair enumeration.

Scope predicate: `p.depths_nondecreasing and p.weights_nonnegative and p.rounding_admitted`.

- Precondition `pre_0` (semantic): `p.depths_nondecreasing and p.weights_nonnegative and p.rounding_admitted` — Absolute depth difference can be replaced by ordered subtraction; ordering is semantic.

## Local example and misuse

For a local instance of `map`, identify the object `ordered_pair_distance_prefix_moments` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/gsplat/delta/ordered_pair_distance_prefix_moments@1/template.yaml`. Original outline: `knowledge/studies/gsplat/delta/ordered_pair_distance_prefix_moments@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: O(1) state and O(n) work instead of pair enumeration.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `pair_expansion` (proof): At prefix j, D equals twice sum_i<k<=j w_i*w_k*(z_k-z_i); A,B are prefix moments.
- `ordering` (proof): Without sorted evaluated depth, signed subtraction cannot replace absolute difference.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
