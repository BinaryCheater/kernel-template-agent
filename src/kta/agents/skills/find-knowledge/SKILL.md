---
name: find-knowledge
description: How to find knowledge in the KTA pool with kta query and kta show, and how to judge whether a retrieved Template, chain or family applies to your case. Read it in every task that starts from existing knowledge.
---

# Finding and judging knowledge

The pool holds Patterns (kinds of transactions), Templates (one refinement step each),
Chains (a sequence of steps for one workload), Families (an executable chain with a
generator), References (frozen external kernels) and hardware facts. You reach it only
through `kta query` and `kta show`; every lookup is recorded, so look up what you use.

## Search in this order

1. **Nearest chain.** `kta query chain --near <case or family>`. A chain that already
   solves a close workload is the cheapest start: you change only the steps whose
   conditions fail in your case.
2. **Pool for the case.** `kta query pool --case <id>` lists what applies, with the
   condition status of each item.
3. **Templates by Pattern.** Name the transaction you need to refine (its Pattern), then
   `kta query tmpl --pattern <pattern>`; add words with `--text`. If no Pattern fits,
   search by words and record the gap.
4. **Details.** `kta show <id>` for the Template's variables, preconditions, Γ (context
   conditions), obligations, cost terms and sources.

## Is it applicable?

A retrieved item applies only if all of these hold. Check them; do not assume them.

| Question | Where to look | If unknown |
| --- | --- | --- |
| Does it refine the transaction you have (same Pattern, same level)? | Template `refines`, `level` | treat as not applicable |
| Does Γ hold for every shape of the case? | `kta query pool` shows Γ true/false/unknown | unknown means you must state the assumption |
| Are the hardware features it requires supported on the target? | `requires_hw`, `kta show <HwFeature>` | do not use it |
| Can your step's choices be written as values of its variables? | Template variables and domains | partial match: say what is missing |
| Are its obligations satisfiable here (capacity, ordering, alignment)? | obligations, constraints | name the one you cannot meet |

Status and assurance matter: `declared` was written but not tested; `tested`, `bounded`,
`proved` carry checker evidence. Prefer stronger assurance when two items fit equally.

## Using what you found

- Cite every record you rely on by ID in `NOTES.md` and in the delivery note.
- When an item almost fits, keep the difference explicit (a Gap, or a new Template
  version) instead of stretching it.
- Retrieved cards are summaries written by their authors; confirm a claim with
  `kta show` or `kta why` before building on it.
- Do not copy whole records into your workspace; refer to them by ID.
