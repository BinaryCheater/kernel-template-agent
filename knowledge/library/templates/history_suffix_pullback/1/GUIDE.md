# history_suffix_pullback@1: history suffix pullback

## When to consider

When a streaming operator returns the trailing input history as a differentiable output, route its gradient to both current-input and initial-history coordinates according to the concatenated logical stream.

## Computation before and after

When a streaming operator returns the trailing input history as a differentiable output, route its gradient to both current-input and initial-history coordinates according to the concatenated logical stream.

Let initial history h have W-1 samples at logical indices -(W-1)..-1, then new input x have L samples0..L-1. Returned history f[j] is concatenated_stream[L-W+1+j], j0..W-2. For final-history cotangent df[j], let q=L-W+1+j: add df[j] to dx[q] if q>=0, otherwise dh[q+W-1]. Add these contributions to the independent stencil pullback, do not replace it. This includes short invocations L<W-1, where final history contains part of initial history. Fixed boundary ownership allows direct adds after the local stencil gradient; if multiple producers share a state coordinate use a complete sum protocol.

Recover the original computation: Differentiable suffix history, source widths2–4 including invocation shorter than history. This is independent of the numeric convolution weights and does not imply gradients through discrete sequence ids.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| width | compile / software | {"lo": 2, "hi": 4} | W: source stencil width. |
| new_tokens | compile / software | {"lo": 1, "hi": 2147483647} | L: invocation length, workload input. |

## Coupled constraints and scope

Differentiable suffix history, source widths2–4 including invocation shorter than history. This is independent of the numeric convolution weights and does not imply gradients through discrete sequence ids.

Scope predicate: `(p.returned_history_is_exact_suffix and p.gradient_initialization_defined and p.each_boundary_contribution_owned_once and p.no_early_overwrite)`.

- Precondition `pre0` (semantic): `p.returned_history_is_exact_suffix and p.gradient_initialization_defined and p.each_boundary_contribution_owned_once and p.no_early_overwrite` — Exact suffix convention and complete accumulation into initialized gradients.

## Local example and misuse

For a local instance of `map`, identify the object `history_suffix_pullback` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/causal-conv1d/delta/history_suffix_pullback@1/template.yaml`. Original outline: `knowledge/studies/causal-conv1d/delta/history_suffix_pullback@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `suffix_address` (proof): q=L-W+1+j maps every returned sample to its original stream coordinate.
- `pullback_sum` (proof): Each df appears once at that coordinate and adds to all ordinary stencil contributions.

Floating-point reassociation, casts and approximate functions require an admitted numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
