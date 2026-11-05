# early_dependent_grid_launch@1: early dependent grid launch

## When to consider

When launch overhead or independent dependent-grid prologue can overlap a predecessor epilogue, release scheduling early and retain a completion wait before dependent loads. In vector add, computing offsets can run early; reading x/y written by the previous grid cannot. A launch trigger is never permission to consume unfinished output.

## Computation before and after

A predecessor grid releases dependent launch scheduling before its completion while dependents still join completion before reads.

Predecessor programs issue a launch-dependents trigger after their necessary prologue, potentially before output epilogue stores. Once all programs have triggered or finished, the runtime may schedule the dependent grid if resources permit. A dependent may compute only independent indices/resources before grid-dependency wait; it must execute the completion wait before reading any predecessor-produced value. Triggering alone provides no memory publication. Keep ordinary lifetime/stream completion before final external observation.

Recover the original computation: Move independent dependent-grid prologue work after predecessor completion and erase the early scheduling trigger; all dependent data reads still see the completed predecessor result.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

NVIDIA supported PDL launch chains, CUDA SM90+ source gate. This is early scheduling with whole-grid completion, not per-tile ready publication or a transfer completion barrier.

Scope predicate: `(p.pdl_supported_and_enabled_for_both_launches and p.same_dependency_chain) and (p.every_predecessor_program_triggers_or_finishes and p.consumer_waits_before_dependent_reads) and (p.prewait_work_independent_of_predecessor and p.buffer_lifetime_valid)`.

- Precondition `pre0` (semantic): `p.pdl_supported_and_enabled_for_both_launches and p.same_dependency_chain` — Both grids use supported programmatic dependent launch in the actual stream chain.
- Precondition `pre1` (semantic): `p.every_predecessor_program_triggers_or_finishes and p.consumer_waits_before_dependent_reads` — Scheduling release is not mistaken for data visibility.
- Precondition `pre2` (semantic): `p.prewait_work_independent_of_predecessor and p.buffer_lifetime_valid` — Only independent prologue work may run before the completion join.

## Local example and misuse

For a local instance of `sync`, identify the object `early_dependent_grid_launch` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/triton/delta/early_dependent_grid_launch@1/template.yaml`. Original outline: `knowledge/studies/triton/delta/early_dependent_grid_launch@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `dependency_visibility` (proof): Every predecessor-dependent read follows the whole-grid completion join.
- `release_progress` (test): All required predecessor programs release scheduling or finish without circular dependency.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
