# shifted_innovation_recurrence@1: shifted innovation recurrence

## When to consider

When a decayed state update uses both previous and current innovations, absorb the next-step previous-innovation coefficient into the current insertion and correct the current output separately.

## Computation before and after

When a decayed state update uses both previous and current innovations, absorb the next-step previous-innovation coefficient into the current insertion and correct the current output separately.

Let U_t be an input-dependent innovation, alpha_t decay, eta_t previous-innovation coefficient and gamma_t current coefficient. Original S_t=alpha_t*S_(t-1)+alpha_t*eta_t*U_(t-1)+gamma_t*U_t. Define T_t=S_t+eta_(t+1)*U_t, so T_t=alpha_t*T_(t-1)+(gamma_t+eta_(t+1))*U_t. Seed T_before=S_before+eta_first*U_before. Output L_t(S_t)=L_t(T_t)-eta_(t+1)*L_t(U_t), for linear state projection L_t. Source computes strictly-past transformed contributions plus gamma_t*L_t(U_t), avoiding subtractive correction of a rounded diagonal. Last eta_after=0 returns original final S; continuing a chunk/call must preserve the declared boundary convention or convert explicitly. For outer product U=v outer k, fold gamma+eta_next into k before contraction. Store unscaled k and v if needed by next call.

Recover the original computation: Linear decayed state with two adjacent innovation terms. Mamba3 eta=dt*(1-sigmoid(trap)), gamma=dt*sigmoid(trap), alpha=exp(ADT). No state-dependent nonlinear gate can be moved through this identity.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Linear decayed state with two adjacent innovation terms. Mamba3 eta=dt*(1-sigmoid(trap)), gamma=dt*sigmoid(trap), alpha=exp(ADT). No state-dependent nonlinear gate can be moved through this identity.

Scope predicate: `(p.linear_state_projection and p.coefficients_input_dependent_only and p.shifted_boundary_defined and p.reassociation_admitted)`.

- Precondition `pre0` (semantic): `p.linear_state_projection and p.coefficients_input_dependent_only and p.shifted_boundary_defined and p.reassociation_admitted` — Linear output projection, known coefficients, matching initial/final state convention and numeric admission.

## Local example and misuse

For a local instance of `scan`, identify the object `shifted_innovation_recurrence` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/mamba/delta/shifted_innovation_recurrence@1/template.yaml`. Original outline: `knowledge/studies/mamba/delta/shifted_innovation_recurrence@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `state_translation` (proof): Substitute Tprev=Sprev+eta_current*Uprev to recover original recurrence exactly in real arithmetic.
- `output_correction` (proof): Strict-past transformed history plus gamma_current*current projected innovation equals original projected state.
- `boundary_state` (proof): Seed and returned state convert eta*U at invocation boundaries exactly once.

Floating-point reassociation, casts and approximate functions require an admitted numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
