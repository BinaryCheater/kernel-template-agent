# statistic_dependent_rematerialization@1: transform values across group statistic

## When to consider

When output codes need a row statistic over transformed values, keep the transformed row in registers or recompute it after the statistic join. Recompute reduces live registers but repeats input traffic and transform work; all outputs must still use the same full-group statistic and original inputs. Source ConvRot chooses recomputation forK>32768.

## Computation before and after

Trade transformed-value register residence against an identical second pure transform pass after a group statistic.

Original operation computesz_i=f(x_i), group statistic s=reduce_i h(z_i), theny_i=g(z_i,s). Resident mode retains eachz_i until the completed statistic is visible, then emits output. Recompute mode reducesh(f(x_i)) in a first pass, discards transformed values, joins the statistic, and evaluates the identicalf(x_i) again for output. Preserve the group domain, final statistic and every numerical rounding step. No output may overwrite an input that another second-pass computation still needs.

Recover the original computation: Erase the repeated pure evaluation in recompute mode; identical source generation and transform recipe give the samez_i used by the original output and statistic.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| retention | compile / software | ["resident", "recompute"] | Keep transformed values live through the statistic join, or discard and evaluate the same transform again from immutable inputs. |

## Coupled constraints and scope

Independent transformed coordinates/groups whose output needs a complete enclosing-group statistic. Source ConvRot transforms Hadamard subgroups but quantization scale covers the full row; resident path retains the transformed row, largeK recomputes it. General mutable/racy state transitions are excluded.

Scope predicate: `p.transform_is_pure_and_deterministic and p.source_parameters_immutable and p.group_output_dependency_explicit`.

- Precondition `pre_pure` (semantic): `p.transform_is_pure_and_deterministic and p.source_parameters_immutable and p.same_transform_recipe_on_replay` — Recomputation has no side effects and preserves input generation, operation order and casts.
- Precondition `pre_stat` (semantic): `p.statistic_complete_before_output and p.group_output_dependency_explicit` — Every output uses the completed statistic for its own group.
- Precondition `pre_lifetime` (implementation): `p.recompute_inputs_live_until_last_read and p.output_stores_do_not_destroy_unread_inputs` — A second pass can still read original inputs; all cross-thread reads and barriers have matching ownership.
- Constraint `c_resident` (resource): `retention != 'resident' or p.transformed_live_bytes <= p.register_budget_bytes` — Resident transformed values and required live work fit the admitted register budget.

## Local example and misuse

For a local instance of `map`, identify the object `transform_values_across_group_statistic` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/statistic_dependent_rematerialization@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/statistic_dependent_rematerialization@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: ResidentO(row values) versus bounded tile and reduced-statistic storage.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `replay_identity` (test): Each second-pass transformed value equals the first-pass value under the fixed numerical recipe.
- `statistic_join` (test): Every output uses one complete matching-group statistic, with no stale or partial reduction.
- `lifetime` (test): Inputs and parameters survive until the last recomputation and no output destroys an unread dependency.

Changing transform rounding or using an algebraically different fast path invalidates replay identity. Only the identical pure recipe justifies recomputation equivalence.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
