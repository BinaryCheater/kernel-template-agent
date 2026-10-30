---
name: write-template
description: How to extract one refinement step into a Template record — what it refines, what it introduces, its variables, conditions, obligations and cost terms, with exact source citations — and check it with kta validate-template. Used for task T3.
---

# Writing a Template

A Template describes one refinement step: it replaces one coarse transaction (an
instance of a Pattern) by finer ones and introduces one new object. It says when the
step is valid and what it costs, in terms general enough to reuse on other kernels.

## Where the step comes from

The task names a source (a Reference with frozen commit, or a Gap from a
decomposition). Read only the code that implements the step (`kta src <reference>
<symbol>`); cite every claim as `path:line` at the frozen commit. A claim without a
citation is not admitted.

## Fill the record (spec/schemas.md §4.2 field names; use `inputs/` examples)

1. **What it refines.** The Pattern of the transaction it replaces and the level (1–7:
   workload, graph, operators, tensor map, instructions, architecture, microarchitecture).
   If no Pattern fits, use `other` and describe the transaction precisely.
2. **What it produces.** The finer transactions and the one new object.
3. **Variables.** Each choice the step makes, with role (software, hardware, boundary),
   kind (structure or numeric), domain and binding time (compile or runtime). Name them
   by meaning, not by the source's identifiers.
4. **Preconditions and Γ.** When the step is valid (on Pattern parameters `p.*`) and in
   which context it is worth it (`case.*`, `hw.*`). Use the expression language; one
   condition per fact.
5. **Constraints and obligations.** What every valid assignment must satisfy
   (capacity, alignment, ordering, lifetime), and the local proof obligations (e.g. "a
   read happens after its transfer completes"), each with the method you expect to
   discharge it.
6. **Cost terms.** Per hardware unit, per instance of the step (`p.instances` scales
   them). Only terms you can justify from the code or a cited fact.
7. **Hardware requirements.** `requires` lists abstract Capabilities (e.g. `async_copy`,
   `managed_scratchpad`), never ISA instruction names; leave `probed_on` empty (the
   system fills it from runs). Code fragments and expected SASS go to a Realisation of
   the Template (backend × ISA range), not into the Template. A capability with no name
   yet is described in the scope text.
8. **Scope text.** One paragraph: what the step is and where it does not apply.

The Template's assurance is `declared`: you assert it; checkers raise it later.

## Granularity

- One step, one new object. If a code region changes several transactions at once,
  either split it into several Templates that each make sense alone, or keep one and say
  in the scope text which transactions it refines together.
- Mechanism over kernel: describe the mechanism so it applies beyond the source kernel;
  never name the source kernel as the Template.

## Self-check

`kta validate-template out/template.yaml`: fix every error. `possible_duplicate` must
be answered in the delivery note (why it is a different step, or why a new version).
`unknown_hw_param` and `constraint_false` warnings need a sentence each.
