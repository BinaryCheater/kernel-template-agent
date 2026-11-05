# ordered_affine_prefix@1: ordered affine prefix

## When to consider

When a recurrence has state-independent affine coefficients, replace serial token updates with an ordered associative prefix over affine maps; retain order because the combine is not commutative.

## Computation before and after

When a recurrence has state-independent affine coefficients, replace serial token updates with an ordered associative prefix over affine maps; retain order because the combine is not commutative.

Represent token t by (a_t,b_t) for s_t=a_t*s_(t-1)+b_t. Compose earlier (a0,b0) then later (a1,b1) as (a1*a0,a1*b0+b1). Inclusive ordered scan yields (A_t,B_t), then s_t=A_t*s_initial+B_t. Chunk scans seed with the preceding aggregate; padded tail tokens are identity(1,0). A segment reset replaces its first coefficient a by zero so prior history cannot cross it. Real/complex diagonal coordinates scan independently; nonlinear output projections/gates occur after recovery, not inside this affine scan.

Recover the original computation: Ordered diagonal real/complex affine recurrences; each prefix pair denotes the same function of initial state. General matrix transitions, state-dependent coefficients and nonlinear recurrence are outside this version.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| chunk_size | compile / software | {"lo": 1, "hi": 65536} | Free positive tile extent; scan tree and hardware menu separately constrained. |

## Coupled constraints and scope

Ordered diagonal real/complex affine recurrences; each prefix pair denotes the same function of initial state. General matrix transitions, state-dependent coefficients and nonlinear recurrence are outside this version.

Scope predicate: `(p.coefficients_independent_of_prior_state and p.affine_coordinatewise_state and p.ordered_scan and p.rounding_admitted)`.

- Precondition `pre0` (semantic): `p.coefficients_independent_of_prior_state and p.affine_coordinatewise_state and p.ordered_scan and p.rounding_admitted` — Coefficients may depend on input but not unknown state. Ordered associativity is required, not commutativity.

## Local example and misuse

For a local instance of `scan`, identify the object `ordered_affine_prefix` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/mamba/delta/ordered_affine_prefix@1/template.yaml`. Original outline: `knowledge/studies/mamba/delta/ordered_affine_prefix@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `composition` (proof): Substitution of s1=a0*s+b0 into s2=a1*s1+b1 gives composed coefficients; associative function composition preserves chronological order.
- `tail_and_reset` (proof): Identity padding leaves final state unchanged; zero a at reset annihilates prior-state influence.
- `chunk_carry` (proof): Seed every chunk with its exact previous aggregate before exposing outputs.

Floating-point reassociation, casts and approximate functions require an admitted numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
