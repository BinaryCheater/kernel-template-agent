---
name: write-family
description: How to build or extend an executable family — a chain of Template steps, the generator that emits code for each assignment, and the code markers that tie code regions to chain items — and how to check it with kta validate-family. Used for tasks T1 and T6.
---

# Writing a family

A family is an executable chain: the chain lists the refinement steps (each binds one
Template), and the generator turns an assignment of the variables into kernel source.
The system, not you, enumerates assignments, compiles, checks and measures them.

## Start from what exists

1. `kta query chain --near <case>` and `kta show <family>`: if a close family exists,
   copy its directory from `inputs/` to `out/family/` and change only what the new case
   needs. Bump the version; keep the name.
2. For each chain item, check whether its Template's Γ and preconditions still hold for
   the new case (skill `find-knowledge`). Only the items that fail need work.

## The family directory

```
family.yaml        # Family body: name, version, backend, variables, glue_constraints, dispatch, entry
chain.yaml         # Chain body: workload, items
gen.py             # generate(assignment: dict, case: dict) -> GeneratedKernel
edge_cases.yaml
README.md          # one page at most
```

For every chain item:

- `key`: unique name inside the chain;
- `template`: the Template it applies (`name@version`);
- `applies_to`: the concrete transaction it refines;
- `bindings`: each Template variable -> a family variable, a constant, or an expression
  over family variables, `case.*` and `hw.*` (`p.<name>` for Pattern parameters);
- `after`: keys of the items it depends on (a DAG);
- `code_markers`: names of the marked regions in the generated code that implement it.

In the generated source, wrap the code of each item between its markers, so a checker
can relate code to steps. A region without an item, or an item without a region, is an
error. Every glue constraint's `meaning` must say why it cannot belong to one step.

## Changing one step

1. Say in `NOTES.md` which item changes and why (which condition failed).
2. Pick the replacement Template (or a new version of the same one). If none exists,
   stop: describe the missing step as a Gap in the delivery note rather than writing
   untracked code.
3. Change the bindings and only the marked regions of that item.
4. Re-check the constraints that mention the changed variables.

## Self-check before submit

- `kta validate-family out/family`: chain consistency (every item bound and marked,
  every constraint evaluable) and space verification. Fix every error.
- For a few assignments, `kta instantiate` and `kta build` to see that the generator
  emits code that compiles; read the feedback (registers, spills, shared memory).
- Do not tune by running many candidates yourself; the system does that.

## Delivery note

List the items you changed, the Templates (IDs) each now binds, the conditions that
made the change necessary, every constraint you added, and anything you could not
express (as a Gap description).
