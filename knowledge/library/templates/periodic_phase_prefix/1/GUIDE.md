# periodic_phase_prefix@1: periodic phase prefix

## When to consider

When only a periodic phase is observable through rotation, scan phase increments and reduce modulo the period at chunk boundaries to bound carried magnitude.

## Computation before and after

When only a periodic phase is observable through rotation, scan phase increments and reduce modulo the period at chunk boundaries to bound carried magnitude.

For increments delta_t and initial phase theta, form within-chunk inclusive prefix theta+sum_(j<=t)delta_j, emit modulo P, and carry modulo P of theta+sum_chunk(delta). This represents the same phase on the quotient R/(P Z) in exact arithmetic; all consumers must be P-periodic or explicitly consume the canonical representative. Source P=2*pi, delta=tanh(angle_projection)*pi*dt. Pairwise or split-half rotary layouts must pair the same two coordinates in producer and consumer. A backend that leaves phase unwrapped is a different numerical realization; large-angle accuracy and modulo seam derivatives require their own contract.

Recover the original computation: Periodic phase state, source rotary2π with masked padded increments0. This is quotient-state equivalence, not equality of floating-point cumsum algorithms or raw phase arrays.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Periodic phase state, source rotary2π with masked padded increments0. This is quotient-state equivalence, not equality of floating-point cumsum algorithms or raw phase arrays.

Scope predicate: `(p.period_positive and p.consumers_periodic_or_canonical and p.phase_rounding_admitted and p.chunk_partition_complete)`.

- Precondition `pre0` (semantic): `p.period_positive and p.consumers_periodic_or_canonical and p.phase_rounding_admitted and p.chunk_partition_complete` — Period and consumer contract, admitted floating-point range reduction and complete prefixes.

## Local example and misuse

For a local instance of `scan`, identify the object `periodic_phase_prefix` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/mamba/delta/periodic_phase_prefix@1/template.yaml`. Original outline: `knowledge/studies/mamba/delta/periodic_phase_prefix@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `quotient_prefix` (proof): Adding any integer multiple of P to the carried phase leaves the next emitted phase equivalent.
- `pair_layout` (proof): Rotation maps the intended coordinate pairs and leaves unrotated coordinates unchanged.
- `numeric_seam` (proof): Range reduction, trig approximation and backward derivative are admitted around modulo seams.

Floating-point reassociation, casts and approximate functions require an admitted numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
