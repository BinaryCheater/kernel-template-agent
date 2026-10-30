---
name: decompose-kernel
description: Decompose a reference kernel into a KTA chain — list every transaction it performs on the hardware, write the refinement steps from the workload down, match existing Templates, and record unmatched transactions and Gaps. Used for task T2 and for manual decomposition (WP10a).
---

# Decomposing a reference kernel

Goal: answer "can this kernel be written as a composition of existing Templates, and if
not, what is missing?". You produce evidence, not an assessment. Mark uncertainty as
such; never force an existing Template onto a step to make the chain look complete.

This skill describes a method only. It contains no conclusion about how any particular
kernel should be designed.

## Inputs

- The Reference record: source frozen to a commit, build recipe, whether it runs on the
  current Target.
- If it runs: existing Measurements, SASS (`kta sass`), profile (`kta profile`).
- Source access: `kta src <reference> <symbol or words>`; do not read the whole tree.

Decompose one configuration (one dtype, one code path). Name the configuration, and the
compiled-out variants you exclude, in the delivery note.

## Step 1: the workload

One paragraph and one table: input and output tensors (shape parameters, dtype,
layout), the mathematical definition, the allowed error, runtime parameters (for
example a table of sequence lengths). Only *what* is computed, not *how*.

## Step 2: the transaction inventory

Start from the source; use SASS and profile to confirm. List every transaction by class,
each with its source location (`path:line` at the frozen commit):

| Class | What to find | Clues in the source | How to confirm |
| --- | --- | --- | --- |
| compute | every reduction, contraction, elementwise op; which unit and instruction | accumulating loops; matrix-multiply calls; math functions | kinds and counts of compute instructions in SASS |
| storage | where each intermediate lives (registers, shared, other on-chip, global), layout, how many copies | array and shared declarations; allocations | registers and shared memory in the build feedback |
| transfer | between which levels; synchronous or asynchronous; when issued, when waited; granularity | loads/stores; copy calls; async copies and waits | memory and async-copy instructions in SASS |
| synchronization | every barrier or wait and what it protects | barriers; fences; wait calls | synchronization instructions in SASS |
| executors | which threads, warps, CTAs do what; role specialization | branches on thread or warp index | — |
| scheduling | loop order; tile traversal order; persistent or not; runtime work assignment | grid computation; work queues; metadata | — |
| numerics | changes of the mathematical form or precision (approximations, scaling, low-precision accumulation) | special-function code; scale factors | error against the reference result |
| compiler-decided | register allocation, unrolling, and any lowering the source leaves to the compiler | no explicit code | build feedback, SASS (record as one transaction per kind, attribution `intent_sass`) |

The inventory must be complete: step 4 checks the chain against every entry.

## Step 3: the refinement chain, top down

Start from the workload. Each step answers one question: **which coarse transaction is
replaced by which finer transactions here, and what new object does it introduce?**

For each step:

1. Decide its level (0 graph, 1 operator transformation, 2 tensor execution mapping,
   3 instruction scheduling, 4 system architecture, 5 microarchitecture).
2. Decide the Pattern of the refined transaction: `kta query tmpl --pattern <pattern>`.
   If no Pattern describes it, use `other` and describe it precisely.
3. Retrieve Templates that refine that Pattern: `kta query tmpl --pattern <pattern>
   --text <words>`. Read each card; `kta show <id>` for variables and Γ (skill
   `find-knowledge`).
4. Judge the match:
   - **match**: the step's new object is of the kind the Template produces, every
     choice of the step is a value of the Template's variables, and Γ holds in this
     kernel's context. Write a chain item with the bindings (`p.<name>` for Pattern
     parameters), empty `code_markers` (the reference has none of our markers), the
     concrete transaction in `applies_to`, and the source lines.
   - **partial match**: mostly the same, but a choice the Template cannot express. Do not
     force it. Write the chain item *and* a Gap (missing "template" or "relation")
     stating the difference.
   - **no match**: write a Gap describing the step: level, refined Pattern, new object,
     choices introduced, and what you think its local obligation is (for example "a
     read happens after its transfer completes").
5. Write the step's dependencies on earlier steps (`after`).

One step introduces one object. If one code region changes several transactions at
once (for example the executor split and the transfer mechanism), first try to write
them as separate steps; if they cannot be separated, write one step and say in the Gap
that it refines several transactions together, and which.

## Step 4: check the chain against the inventory

Go through the inventory of step 2. Every transaction that no chain item explains goes
into `unmatched_transactions` (level, Pattern guess, description, source location), each
with exactly one Gap.

Counting (so match rates compare across references): `total` is the number of chain
items; `matched` counts the items whose Template existed before this reference and match
without a Gap. Report the unmatched transactions next to the rate: a full match rate with
unexplained transactions is not full coverage.

## Step 5: reproduce (only if it runs and every step matches)

Find the family whose chain is closest (`kta query chain --near`), generate the kernel
with these values, check and measure it, and compare with the reference. Write the
reproduction error into the Reconstruction.

## Output

In `out/`, one record body per YAML file:

```
reference.yaml        # if not yet in the pool
chain.yaml            # spec/schemas.md §4.3
reconstruction.yaml   # spec/schemas.md §4.5
gaps/*.yaml           # one file per Gap
patterns/*.yaml       # proposed Patterns (status proposed)
templates/*.yaml      # proposed Templates (spec/schemas.md §4.2, assurance declared;
                      # write the obligations, no proof needed; skill write-template)
```

Hardware requirements use only feature names from the target's feature list; a feature
without a name there stays out of `requires_hw` and is described in the scope text.

The delivery note gives: the match rate and the unmatched transactions by class; the
three judgements you are least sure of, with reasons; comments on the Pattern table or
Template granularity.

## Common mistakes

- Cutting steps along functions or code blocks. Steps follow "which transaction is
  refined"; one function may hold several steps, several functions may make one.
- Using the reference's name as a Template. A Template describes a refinement step; its
  scope is written with Pattern parameters and the hardware contract.
- Skipping step 4. Without checking the inventory you cannot know the chain is complete.
- Claims without source lines. A judgement without a citation is not admitted.
