# Predicting and searching kernel implementations before compilation

## Research question

Can KTA use a library of Template transformations, a reusable hardware model,
and a small amount of compiler and GPU evidence to select good kernel
implementations before compiling most of the search space? The result should
include a reproducible implementation, a performance prediction, and the full
cost of getting both. Hardware investigation serves this prediction task.

There are two linked questions. First, can the Templates extracted from a
studied production kernel help reconstruct its computation at competitive
performance? Second, can executable versions of those Templates define a
search space whose costs can be predicted and whose solutions can be
materialized deterministically? The first experiment tested coding assistance.
The next experiment should test the second question.

## What the first experiment established

The controlled replay used the pinned FlashAttention-2 CUDA causal FP16 D128
forward kernel as the direct comparator on GB10, with sequence lengths 512,
2048, and 513 and checks for both output and log-sum-exp. The same original
kernel was timed alongside each candidate. The three conditions supplied
18 Templates, offered a 320-entry pool, or supplied zero Templates. Three
independent sessions per condition implemented the same operator. The pool size
counts knowledge entries.

| Information given to the model | Correct | Stable timing | Reached the original-kernel target |
| --- | ---: | ---: | ---: |
| Prescribed Templates | 3/3 | 3/3 | 0/3 |
| Model selects Templates | 3/3 | 2/3 | 1/3 |
| No Templates | 3/3 | 2/3 | 1/3 |

The target allowed up to 11.1% extra GPU time on every shape in every accepted
final window. One additional self-selection run showed roughly 18%, 6%, and
24% lower times in two valid windows and remained unresolved under the
three-window rule. The experiment establishes that both routes can reach the
target. It leaves the efficiency benefit of written Template guidance unresolved.

Agents chose concrete parameters and iterated on measured timings. The
campaign contains 111 submissions, 81.4 minutes from first session to final
measurement excluding prior preparation, and a $11.373 API-equivalent model
cost estimate. The earlier exploratory PyTorch entrypoint results answer a
different question. See [controlled findings](controlled-replay/conclusion.md)
for per-shape microseconds, time and token costs, and protocol deviations, and
[figures](controlled-replay/figures.md) for variability and iteration history.

The consolidated library preserves 312 paired Template and proof-outline
entries, provenance, and unresolved duplicate or incompatible contracts.
It provides a starting point for executable bindings. See the
[library report](template-consolidation-report.md).

## What is missing between Template and MIP

A mathematical graph specifies the computation. A usable **parameterized
implementation** also fixes how the selected transformations affect work
partition, thread roles, memory layout, lifetimes, synchronization, and
pipeline overlap. It declares which decisions can vary and how they compose.
One version of these bindings should drive both the emitter and the precompile
execution description. An agent can help author it once. A fixed assignment
then generates source and launch metadata by ordinary code.

Compiler decisions form a second mapping. A fixed generator does not tell us
the final register allocation, spill traffic, inferred layouts, actual
instruction paths, or inserted waits. Start with a TileLang or CUDA path that
exposes the performance-sensitive choices. Derive what follows from the
implementation, compile a few representative configurations and resource
boundaries, and inspect the resulting instructions and resource metadata.
Use GPU timing for remaining performance terms that the model cannot explain
from those observations. Treat family-specific fitted terms as software-model
corrections with a recorded scope. Keep them distinct from hardware parameters.

L3 evaluates a compiled implementation. Its mechanism rules and compiled
witnesses help build and check L2. A concrete SASS trace alone does not specify
how instructions would change under another tile, pipeline, or layout choice.
L2 should preserve the resource usage and execution dependencies that change
the decision while aggregating finer instructions. Its useful accuracy is an
empirical property of a specific family, toolchain, and parameter range.

Use the current L3 as the starting point for the pilot. Its round-20 blind
report finds the measured best in the predicted top three for 8 of 11 groups.
Validate selection against real timings on the new family and investigate L3
errors when they change a decision in that family.
[Performance model state](../../design/perfmodel/STATE.md).

MIP should be generated mechanically from supported symbolic constraints and
costs. Fixing some parameters is straightforward. Converting general
resource scheduling, cache behavior, or compiler transitions into a compact
linear formulation can require approximations. A practical first design
uses a limited projection to propose candidates, L2 to rank them, and
compilation and GPU evaluation to confirm the strongest choices. Keep a
proven bound separate from an estimated cost. See the
[detailed design and open obligations](model-search-pilot.md).

## What a successful next result looks like

1. A working, fixed implementation family with meaningful legal choices.
   The same bindings change both code and execution description. Repeated
   generation of an assignment produces the same source and launch description.
2. A small offline path that produces and solves a real MIP and compiles,
   checks, and times the generated kernel on GB10. Verify the MIP translation
   against direct evaluation of the same expression on a small legal domain.
3. Separate evidence for three gaps. Compare the precompile description with
   the compiled one, the L2 execution prediction with L3 and GPU results, and
   the MIP projection with L2. Reserve configurations or shapes from
   calibration and freeze predictions before evaluating them.
4. A comparison with empirical tuning in the same family under comparable
   budgets. Report the slowdown from the fastest measured point, whether
   the shortlist retained that point, and how many candidates required
   compilation or timing. Retain absolute prediction errors and failures.
   Keep the exact production kernel as the performance reference.
5. A cost account covering authoring, token usage, compilation, simulation,
   solving, and GPU time. Separate first-use cost from reuse on new shapes.
   If prediction needs nearly as many measurements as direct tuning, the
   cost comparison should show it.

Start with one retained attention implementation and a bounded family.
A small measured validation subset can provide a known best within that subset.
Once the flow works, use an already studied quantization, fusion, or irregular
boundary case to test transfer. Keep parameter search within a family separate
from the later question of choosing and composing Template chains. The
[Claude handoff](claude-handoff.md) gives the starting instructions and data paths.
