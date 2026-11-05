# diagonal_dependency_wavefront@1: diagonal dependency wavefront

## When to consider

When each grid state depends only on its north and west neighbors, expose anti-diagonal parallelism and retain the predecessor frontier rather than serializing the whole grid.

## Computation before and after

When each grid state depends only on its north and west neighbors, expose anti-diagonal parallelism and retain the predecessor frontier rather than serializing the whole grid.

For grid(T,U), initialize required boundary state. At step d evaluate all valid cells(t,u) with t+u=d, using only frontier d-1. A CTA assigns u=tid+k*threads, t=d-u and synchronizes after every diagonal. For source log-probability DP, state=logaddexp(north+blank_edge,west+label_edge); first row/column have one predecessor. Reverse DP initializes final blank transition and traverses decreasing diagonals. A batched-load realization prefetches K edge values per thread but advances dependent states one diagonal at a time. Store frontier in two shared buffers or retain same-u predecessor in a register and read neighbor-u from prior shared frontier; do not overwrite a frontier while a peer can still read it. Packed inputs alter edge addresses only, not the recurrence.

Recover the original computation: Two-neighbor acyclic rectangular dynamic programs with declared boundary/terminal rules; source one CTA per sequence and direction. All-negative-infinity logaddexp needs a separately guarded implementation.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| prefetch_steps | compile / software | {"lo": 1, "hi": 32} | Independent edge prefetch batch K; not a simultaneous DP advance. |

## Coupled constraints and scope

Two-neighbor acyclic rectangular dynamic programs with declared boundary/terminal rules; source one CTA per sequence and direction. All-negative-infinity logaddexp needs a separately guarded implementation.

Scope predicate: `(p.dependencies_in_previous_diagonal and p.boundary_states_defined and p.frontier_visibility_and_lifetime and p.logaddexp_domain_admitted)`.

- Precondition `pre0` (semantic): `p.dependencies_in_previous_diagonal and p.boundary_states_defined and p.frontier_visibility_and_lifetime and p.logaddexp_domain_admitted` — Acyclic north/west dependencies, correct boundary/terminal states, complete frontier synchronization and numerical domain.

## Local example and misuse

For a local instance of `scan`, identify the object `diagonal_dependency_wavefront` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/apex/delta/diagonal_dependency_wavefront@1/template.yaml`. Original outline: `knowledge/studies/apex/delta/diagonal_dependency_wavefront@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `topology` (proof): Both parents have diagonal d-1, so all current-diagonal cells are independent.
- `frontier_lifetime` (proof): All reads of previous frontier complete before its next overwrite.
- `boundary` (proof): Single-parent boundaries and terminal blank contribution recover the requested path sum.

Floating-point reassociation, casts and approximate functions require an admitted numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
