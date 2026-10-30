---
name: design-interrogation
description: Interrogate a mechanism, process or spec before and after designing it, so that defaults, assumptions, failure modes and control knobs are stated instead of silently chosen. Use whenever you design or change how the system works, and at the end of any design conversation.
---

# Design interrogation

## Why this skill exists

An assistant takes defaults at every step and names only the ones it is asked about. On a complex system this is how surprises are made: a silent fallback, an unstated assumption, a mechanism with no way to observe its failure. The method below is not new; it is the standard discipline of systems engineering (requirements, interfaces, verification), failure-mode analysis, control loops (observe, actuate), design rationale (question, options, criteria) and the scientific method (claim, evidence). The skill makes the agent apply it unprompted.

## When to use

The pass pays for itself only on decisions that other work will build on. Run it when one of these is true:

1. A change touches an interface or schema in `spec/` (`schemas.md`, `modules/*.md` signatures, command names, record fields).
2. A new loop, state, task type, process, role, document or bridge object is added.
3. A threshold, budget or policy that governs agent behaviour is introduced or changed (anything that would be a row in `docs/design/12-agent-controls.md`).
4. A decision is about to be recorded as `needs user review`, or a delivery note is about to declare a fixture or simplification.
5. A design conversation ends and its record is about to be written.
6. A milestone closes: the adversarial second pass runs then, in a fresh context, together with the independent acceptance of 60 §5.

## When not to use

- Implementing an interface that `spec/` already fixes; bug fixes; refactors that keep behaviour; tests; scripts; delivery notes; documentation edits.
- Local, reversible decisions inside one WP: these take one row in the "默认" table of `docs/decisions.md` (what, why, how to change), not the nine questions.

## Scale

- **Full pass** (nine questions + artefacts): triggers 1–5. Budget: under 10 % of the task's time. If it takes longer, the thing being designed is larger than one task and should be split; the pass is not the problem.
- **Adversarial second pass** in a fresh context: trigger 6 only, and before a spec freeze. Not per task.
- **Short form** for anything else that still feels like a choice: three questions, one line each — what did I default, how would its failure show, what was the alternative. If the answers are trivial, no record is needed.
- When a user asks "did you consider X?" and the honest answer is no: run the full pass on that mechanism, not only on X.

## The pass: nine questions per mechanism

Answer every question in writing. "Not applicable" must be argued, not assumed.

| # | Question | What a complete answer contains |
| --- | --- | --- |
| 1 | What is it? | a definition in one sentence; what it is *not* (the neighbours it is confused with) |
| 2 | What claim does it serve? | the research or engineering claim that would weaken without it; if none, it should not exist |
| 3 | What does it depend on? | givens (with evidence) and assumptions (without), listed separately |
| 4 | Where does it hold? | the scope: which inputs, hardware, workloads, phases; what makes it transfer or not |
| 5 | How does it fail? | the failure modes, including the ones an agent would cause: digging in, slacking, gaming the check, silent fallback |
| 6 | How would we know? | the observable for each failure mode: a record, a metric, a canary |
| 7 | What is the knob? | what can be adjusted without code; its default; the failure it prevents; where it lives |
| 8 | How is it repaired? | the loop that corrects it: who detects, who changes what, what is re-verified, what is rolled back |
| 9 | Who decides, at what cost? | user decision or assistant default; alternatives not taken; what reversing it costs later |

## The second pass: adversarial

After the nine questions, run a separate pass that attacks the design. What matters is that the reviewer has not authored the design: the author's context carries the author's blind spots. A fresh context is any of: a new session of the same tool given only the design text; the milestone's independent acceptance by a different model (60 §5), which is the preferred place since the pass runs only at milestones anyway; or the user reading with the checklist below. A sub-agent is one way to get a fresh context, not a requirement, and the repository prefers not to add agents where a session or a person serves (60 §1).

- List every choice the design makes that the user did not make. Each becomes a ledger entry.
- For each assumption, state what breaks if it is false, and which components must change.
- For each check or threshold, state how an agent could pass it without doing the intended work.
- For each loop, state its bridge objects: what it reads from and writes for the neighbouring loops; nothing crosses otherwise.
- List the questions the user is likely to ask that the design does not answer.
- List every place where "we will handle it later" appears, and whether the design records it as an obligation.

## Output

The pass produces three artefacts, in the repository, not in chat:

1. **Ledger entries** for every default: `what was chosen | alternatives | why | change cost | status`. In this repository: `docs/design/01-thinking-model.md` §13 for design-level defaults; `docs/decisions.md` and `docs/review-queue.md` for WP-level ones.
2. **Obligations**: claims the design relies on but has not shown, each with how it will be tested (01 §11).
3. **Knobs**: rows for `docs/design/12-agent-controls.md` for every threshold, budget or policy introduced.

## Rules

- Never fall back silently. If a requested capability cannot be delivered, say so, name the alternative, and record it; do not substitute a simpler version and report the original.
- Name the alternative not taken, every time. A decision without a stated alternative is not a decision, it is a default.
- Prefer a rule to a judgement. Where an agent will act repeatedly, write the rule (threshold, order, stop condition) and let the agent apply it; keep judgement for what rules cannot cover.
- Keep the system as simple as the loops require. A new role, agent, process or document must name the boundary (permission, independence, concurrency) that justifies it.
- Write in English; keep records short; move detail to the document it belongs to and leave a pointer.
