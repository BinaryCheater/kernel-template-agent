---
name: use-kta
description: How to work inside a KTA workspace — the kta commands, what each returns, how to deliver with kta submit, and what you must not do. Read it in every task.
---

# Using KTA

KTA is the only way you reach knowledge, compilers, GPUs and the store. Every `kta` call
is recorded against your task (the `KTA_TASK_TOKEN` in your environment identifies it;
it stops working when the task ends). There is no other channel: no ssh, no direct GPU
use, no reading KTA's data directory.

## The workspace

| Path | Who writes it | What it is |
| --- | --- | --- |
| `AGENTS.md` | system | fixed rules (the same in every session) |
| `TASK.md` | system | this step: goal, inputs, done checks, budget |
| `NOTES.md` | you | your running notes; the next session may start from them alone |
| `inputs/` | system | files copied in for this task (sources, family directories) |
| `out/` | you | your deliverables and `out/DELIVERY.md` |
| `.agents/skills/` | system | skills (method descriptions) |

Changing anything but `NOTES.md` and `out/` is recorded as a violation.

## Commands

Every command accepts `--json`. Large outputs are summarised and stored; the command
prints an ID, and `--full` prints the whole text. Use only the commands `TASK.md` lists
as allowed.

| Command | Use it to |
| --- | --- |
| `kta show <id>` | read one record (Template, Family, Case, Check, …) |
| `kta query pool --case <id>` | list families, Templates and References applicable to a case |
| `kta query tmpl <pattern/tag/words>` | search Templates (see skill `find-knowledge`) |
| `kta query chain --near <case or family>` | find the closest existing chain to start from |
| `kta why <id>` | list the evidence behind a conclusion |
| `kta src <reference> <symbol or words>` | search a reference's source by symbol or keyword |
| `kta instantiate <template> <assignment>` | generate source for one assignment |
| `kta build <source>` | compile; returns registers, spills, shared memory |
| `kta sass <build>` | instruction statistics, compared with expected SASS features |
| `kta check <build> --contract <id>` | numeric check, sanitizer, anti-cheat; returns verdict and failing samples |
| `kta run <build>` | one quick run; not a formal result |
| `kta measure <build>` / `kta profile <build>` | formal measurement / Nsight Compute metrics |
| `kta bound <case>` | theoretical lower bound |
| `kta solve <template> <case>` | candidates and predictions |
| `kta validate-template <file>` / `kta validate-family <dir>` | the checks your deliverable must pass |
| `kta note <text>` | attach a short note to the task record |
| `kta submit` | deliver `out/` with `out/DELIVERY.md` |

Exit codes: 0 ok, 1 check did not pass, 2 usage error, 3 not supported, 4 infrastructure
error (retry later; it does not count against you), 5 permission denied (do not retry
around it).

## Case tasks: the header and the access discipline

A case-loop task's `TASK.md` starts with a one-screen header: the case, its bound, the last
round's gap decomposition, the default FormulationPlan with its score, the budget, and the
**situation** you are in. Fetch what the situation's row says, decide what it says, and
cite the records you used in your plan. Fetching outside the row is allowed: every `kta
show`, `card`, `why`, `query` and `src` is logged against your task with the row it
matched (or none), and the table is revised from those logs.

<!-- access-discipline:begin -->
Access discipline **v1** (your `TASK.md` header names the situation).

| Situation (id) | Fetch | Decide |
| --- | --- | --- |
| start (`start`) | nearest chain with each step's Γ evaluated | most steps hold → start from it, fetch alternatives for failing steps; else diagnostic retrieval |
| diagnostic retrieval (`diagnostic`) | bottleneck unit → patterns → templates ranked by evidence (top N cards) | moves that address the dominant transactions |
| each open variable (`open_variable`) | findings that pin it; the wall bounding it | fix with citation, or keep the menu |
| MIP too large (`mip_too_large`) | structure table, regimes for this workload/chip | move outward or decompose, citing the structure |
| non-algebraic effect (`non_algebraic`) | tabulated effects for the operations involved | use the table or request one over ≤ 2 variables |
| last round was ε (`eps`) | nothing from the space | hand to model diagnosis |
| last round was L_rep with an idle resource (`idle_resource`) | templates reducing or overlapping that resource; references with such a step | open a dimension or add a step |
| bottleneck blocked by a wall (`wall`) | shadow price; templates reducing the walled resource | change the move or record a co-design finding |
| stop (`stop`) | — | every non-default decision cites evidence, or budget reached |
<!-- access-discipline:end -->

## Working rhythm

1. Read `TASK.md`, then `NOTES.md`. Write your plan into `NOTES.md` before you start.
2. Start from the input records; fetch more only when you need them.
3. Do not run candidates one by one to search a space: the system compiles, checks and
   measures candidates in bulk after you submit. Your job is the generator, the
   conditions and the cost terms.
4. Validate locally with `kta validate-*`; fix every error; answer every warning in the
   delivery note.
5. Update `NOTES.md` (what you did, what failed and why, what is next), write
   `out/DELIVERY.md`, run `kta submit`.

## When a submission is rejected

You get a message with the failing checks. Read it, record it in `NOTES.md`, fix only
what failed, and submit again. If a check looks wrong, say so in the delivery note and
`kta note`; do not work around it.

## What you must not do

- Write outside the workspace, or change `AGENTS.md`, `TASK.md`, `inputs/`, `.agents/`.
- Use a GPU, a compiler on a remote machine, ssh or the network directly.
- Read or guess hidden shapes, checker internals or seeds; imitate or bypass a checker.
- Return hard-coded numbers or fake outputs; hide a simplification.
