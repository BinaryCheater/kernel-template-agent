# 01 Thinking model

**What this is.** The way we think about KTA as a whole: what we want to show, how we think about the problem, what we are careful about, what has to exist, what we have decided and still must decide, and which experiments settle the rest. It is written to be re-read by a person before any design discussion, so that the questions we cared about and the judgements we made are recoverable without the conversation that produced them.

It records thinking, not detail. Detail that construction needs is in [02 Design notes](02-design-notes.md), organised by the design document it will be merged into; tunable values are in [12 Agent controls](12-agent-controls.md); what each conversation changed is in [the change log](../changelog.md). The paper we build on is `docs/references/2610.39380-learning-to-formulate.pdf` (Formulaic); KTA keeps its template tuple, refinement levels and division of work, and need not match it elsewhere.

**Status.** A living hypothesis, revised by running the system. Nothing here overrides `spec/` until a decision is confirmed and the spec changed.

---

## 1. What we want to show

Designing kernels with an agent can be **white-boxed**: the knowledge an expert uses can be made explicit, checked and accumulated, so that a search over it reaches state-of-the-art kernels, and the agent is needed only where something is genuinely missing. The alternative, a strong model spending many tokens per kernel with nothing kept but code, is the control arm, not the method.

The claim is relative and scoped: for a stated class of workloads, one architecture class and a stated correctness contract. We never claim to cover "all designs". Coverage of SOTA is reported in four levels, and the thesis lives at the third:

| Level | Meaning |
| --- | --- |
| 1 expressible | the SOTA kernel is a point of our space |
| 2 reproducible | generating from that point reproduces its performance |
| 3 reachable | given only the problem, the system finds that point or an equally good one |
| 4 surpassed | the blind result beats the reference under the same contract |

If we can reliably reach SOTA designs, surpassing follows from two things SOTA teams do not do: combining moves across kernels (the union of all references' tricks is larger than any one kernel's), and solving per case for its actual shapes. It must be measured under the same correctness contract, or it is not a result.

Secondary claims: **amortisation** (cost per new case falls as the pool grows; a weak model with the pool approaches a strong model without it) and **diagnosability** (every shortfall is attributed to one component). The evidence so far, with its limits, is in 02 §1.

## 2. How we think about the problem

**One paradigm.** Every kernel is a set of transactions (compute, movement, synchronisation) scheduled on finite resources over time. One vocabulary therefore suffices: a table of transaction patterns and a table of resources, both open. What we cannot describe goes to `other` and is counted; we never pretend the tables are complete.

**One ruler.** Each resource has a measurable peak, so for any problem we can compute the speed-of-light bound per unit without knowing the best kernel. A kernel is finished when some bottleneck unit does no more than the minimal work *and* is busy from start to finish, not merely saturated for an instant. The gap between a result and the bound is decomposed into named parts (inflated work, idle time, a wall in the way, model error, the irreducible) and that decomposition, not the agent's judgement, decides what happens next. Idle time blamed on a wall is the wall's shadow price: on a fixed chip it tells scheduling problems from hardware limits; in co-design it is what wall relaxation buys. This is how kernel optimisation and architecture exploration meet.

**Three kinds of knowledge.** What can be built (templates: the moves; chains: known-good compositions), what it costs (the hardware model), and where it pays (findings, with their scope). The formulation for a case is *derived* from these three; nobody writes a MIP by hand. A SOTA kernel mostly teaches a sentence (a chain with bindings and the regime where it wins), rarely a letter (a new template). Composition knowledge, which moves go together and in what order, is the thing that makes SOTA reachable rather than merely expressible, and it is statistics over reconstructed chains, not something anyone authors.

**The template.** A trick with a contract: how the program changes, when it is safe and why, what it does to cost. It carries no hardware numbers and no performance conclusion; it says what transactions it adds and what may overlap, and the hardware model prices that. One template introduces one object and owes one local obligation. All templates together form a grammar; a kernel family is one sentence of it; the real work of "formulating" is choosing which sentence and which of its variables to leave open.

**The hardware model.** Structured data, not free code, so that the simulator, the MIP and the profiling report all derive from the same source and the simplification from model to MIP is a compiler's job, checkable against the simulator without a GPU. Walls (what cannot be crossed) and rates (how fast each unit is). Fine-grained sharing (two tensor-core paths behind one adder tree) is a resource, discovered by interference tests. The SASS level is the complete substrate, because the instruction stream is the hardware's contract; completeness is never asserted, only measured as coverage. Four fidelity levels share one parameter table: an analytic steady-state projection inside the MIP, a transaction-level simulation that draws the occupancy timeline, the SASS-level simulator we already have, and the hardware.

**Loops with bridges.** Solve, case, knowledge, system; and outside them the design loop that the user and the assistant run. Each loop changes one kind of thing and reads the outer loop's objects as constants; neighbouring loops communicate only through records. Solving workloads, accumulating knowledge and calibrating the model are independent processes that never share a context.

**Levels and transfer.** The refinement levels (1 workload … 7 microarchitecture) say how fine a decision is; the *dependence degree* (semantic, mechanism, realisation, chip, chip × workload) says how far it transfers. Knowledge is attached at the most general layer at which it still holds, so a chip change keeps the semantic and mechanism layers, replaces realisations and instance numbers, and demotes findings to priors. Mechanisms we cannot run on the chip in hand are recorded anyway, marked unprobed, and wake up when a chip that has them arrives.

## 3. What we are careful about

- **We cannot design it all before running.** Only repeated rounds surface the factors a design misses. So the design is a hypothesis, the release is staged (deterministic golden path → hand-stepped agent tasks → one autonomous case loop → knowledge loop → open), and no stage opens until we understand and have debugged the previous one.
- **Defaults are everywhere.** An assistant takes a default at every step and names only the ones it is asked about. Every default is therefore written down with its alternative and change cost (02 §12), and the construction skill `design-interrogation` makes the pass unprompted, with narrow triggers so it does not slow construction.
- **Context is finite.** The failure is drowning, not hiding: pushing everything buries the few items that matter. Agents get a one-screen header and an access discipline (situation → fetch → decide), pull the rest by ID, and the pulls are logged so the discipline can be revised from use.
- **Agents game checks.** Left alone they chase gains of a few per cent that reveal no mechanism, widen search instead of explaining a gap, or polish a case that should be closed. A gain counts only if it is beyond the noise band *and* attributed to a named mechanism; two null rounds close a case; checks are fixed in the task card; hidden shapes stay hidden; planted faults test the checkers.
- **Keep it simple.** One runtime role, a few instances, no sub-agent trees, no multi-agent orchestration. A new role, process or document must name the boundary that justifies it.
- **Agent-hours are the scarce resource; GPU-seconds are cheap.** Learn from measurement rather than reading where possible; decompose at low cost for everything, reproduce only where a new capability is at stake, ablate automatically on our own families, run the full blind loop only for the claims.
- **Set parameters by small experiments, not by fiat.** Which model per task, how much effort, resume or fresh, how many retries, what counts as noise: each is an experiment (02 §11). Until it has run, the value is a default and says so; parameters fixed by discussion hide errors we cannot see.
- **Be honest about verification.** Proofs cover what they cover; everything else is `tested`, never `proved`. The weakest link is whether the agent-written generator implements the template's intent, so SASS traces, sanitizers and later IR-level checks guard it.
- **Knowledge must outlive the chip.** GB10 is not mechanism-complete and we do not have a full Blackwell card. Templates require abstract capabilities, realisations are kept apart, and mechanisms we cannot probe are still recorded so that nothing is searched for twice when the chip changes.
- **Do not let the tools' limits masquerade as missing knowledge.** When the expression language cannot state a condition, that is a gap of the language: counted and admitted at `proposed`, not refused.
- **Extraction need not be right, but it must be checkable.** A decomposition is judged by what follows: can we reproduce it, does another case reuse its steps, does the blind loop find it. Wrong decompositions show up there, not in anyone's opinion of the "right pattern".
- **Coverage is for the complete experiments.** Day-to-day work is forward and "enough for now", as it always was; coverage statistics are for the batch that makes claims about the pool.
- **Structured data serves co-design.** A hardware model that is data can have its walls moved and priced; a simulator that is code cannot. The same object serves kernel optimisation on GB10 and architecture search on Ventus.

## 4. What has to exist

| Component | Why | Designed in | Today |
| --- | --- | --- | --- |
| pattern and resource tables | one vocabulary | 22 §2.2, 02 §4 | patterns seeded |
| template with contract and effects | the unit of a move | 22, 02 §3 | schema exists; `cost` still per unit |
| chain, family, generator | a sentence and its code | 22 §3 | validators exist; first real family pending (WP7) |
| capability and realisation | transfer across chips | 02 §2 | HwFeature only |
| hardware model as declarative data; four fidelity levels | checkable simplification, timelines, co-design | 02 §4–5 | flat peaks + a scalar efficiency; SASS simulator exists outside KTA |
| steady-state projection compiler; MIP↔simulator check | mechanical, checked MIP | 02 §5 | not built |
| gap decomposition and routing | the ruler decides | 02 §6 | partial judging rules |
| formulation plan, deterministic default, chain enumerator | agent choices scored | 02 §5.3–5.5 | not built |
| access discipline and task header | finite context | 02 §5.4, 10 §4 | header designed; discipline not |
| knowledge admission, promotion, invalidation, pool diff view | accumulation you can see | 21, 23, 02 §2 | admission exists; promotion engine and views not |
| learning loop runner (intake, decompose, admit, reproduce, ablate, cover) | automated accumulation | 02 §8 | hand decomposition only |
| regression set and model scoreboard | model you can trust | 02 §4 | calibration code without data |
| canaries, claims ledger, golden path, controls table | supervision without reading code | 01 §3, 12 | 12 written; others not |
| four loops, bridges, staged release | traceable iteration | 02 §6, 10, 11 | processes designed; not running |

## 5. Decisions

**Decided by the user (2026-10-31).** Backends: TileLang primary, CUDA C++ with inline PTX as the escape hatch; CuTe DSL is of limited value on GB10 and is not pursued unless a target needs it. Documents in English. Controlled runs before any autonomous token spend. One tunable table of agent controls, with a settings page later. Parameters set by small experiments. This document stays short and readable; details go to 02 and then to the numbered documents.

**Waiting for the user** (02 §12, "open"): one status ladder for empirical knowledge; how to admit the seven reference decompositions (rejection loop plus counted language gaps, or hand fixes); ablation on our families and coupling as a first-class relation; `knowledge/seed/` as seed only; whether the template-effects and hardware-model changes land before or after the GEMM golden path; learning scope v1 and the reproduction tolerance; blind reconstruction as the learning loop's acceptance experiment.

**Defaults let stand.** Everything else in 02 §12 (D-T1 … D-T30) was chosen by the assistant and is marked as such. Pull on any of them; what is not pulled stays a default and is labelled as one wherever its consequence appears.

## 6. Experiments

**Research obligations.** Each is an assumption the system depends on or a claim it makes; each has a test. O3 (the model ranks) and O5 (attribution is right) come first, because if either fails the loop does not close.

| | To show | Test |
| --- | --- | --- |
| O1 | the vocabulary of moves saturates | ~20 references; new templates per reference → 0; `other` consumed |
| O2 | coupling between steps is local | glue constraints per family; generator diffs confined to markers |
| O3 | the model ranks candidates | rank correlation on top-64 per case; residual per resource |
| O4 | template scope is reliable | rate of "Γ admits, measurement fails" |
| O5 | gap attribution is right | fault injection: degrade the model, delete a template, cap the solver; the router must name the class |
| O6 | agent cost per template and family falls | cost vs pool size |
| O7 | amortisation | learning curve; weak + pool vs strong + scratch |
| O8 | the bound is tight enough | share of unexplained gaps; critical-path model for latency-bound cases |
| O9 | template granularity is right | always-adjacent pairs; single-use templates |
| O10 | the access discipline neither drowns nor starves the agent | discipline vs dump vs free browsing; row ablation |
| O11 | resuming within a work unit beats fresh sessions | two arms: tokens, latency, outcome, rounds stuck |
| O12 | SOTA is reachable blind | give only the case; distance to the reference chain; performance ratio; levels 1–4 |

**System-parameter experiments** X1–X9 (02 §11): model per task type, effort level, resume vs fresh, access discipline, retries, the noise band, reproduction tolerance, measured points and exploration share, task budgets.

**First controlled runs, in order.**
1. S0: the GEMM golden path on GB10 with no agent; numbers against cuBLAS and the bound; every record visible.
2. L1 on the seven references: measure under our protocol, SASS instruction census.
3. L2 on the seven references: reproduce each performance core from its chain; classify failures by step, model or level. This is the cheapest decisive test of how hard extraction really is, and it decides how far to scale source intake.
4. MIP ↔ simulator consistency on the GEMM family, once the projection compiler exists.
5. O12 blind reconstruction on an evaluation set, once the case loop runs autonomously.

## 7. Scope v1 (proposed)

Workloads: GEMM, quantised GEMM/GEMV, attention prefill and decode, MoE grouped GEMM, norm. Target: GB10 (sm_121). Knowledge in scope: scheduling moves inside a kernel. Out of scope and counted separately: system-level features (host planners, paged-KV machinery, PDL, dispatch), graph-level optimisation. Correctness: each reference's own documented tolerance. Sources: the seven decomposed references plus the strongest baseline for each in-scope case; papers enter as claims and queue for reproduction; vendor documents fill the capability table and the hardware model.

## 8. Maintenance

This document is updated when the thinking changes, not per work package, and stays within about 250 lines. It records questions, judgements and decisions; mechanisms go to 02 and from there into the numbered documents; values go to 12; what changed and when goes to the change log. Obligations that are settled move to the claims ledger with their evidence.
