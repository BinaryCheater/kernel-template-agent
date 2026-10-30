# Change log

One entry per design conversation or merged work package. Each entry says what was added or changed, which design requirements came out of it with their status, and what construction should do next. Decision IDs `D-Tn` refer to `docs/design/02-design-notes.md` §12.

## 2026-10-30 → 2026-10-31: thinking-model conversations

### Files added

- `docs/design/01-thinking-model.md`: the human-readable thinking model (claims, mental model, concerns, components, decisions, experiments, scope).
- `docs/design/02-design-notes.md`: detailed material from the conversations, organised by the design document each part merges into; the full decision ledger (D-T1 … D-T30) and open decisions (D-open-1 … 7).
- `docs/design/12-agent-controls.md`: every budget, cut-off, session and anti-gaming control with the failure it prevents and where it is set.
- `.agents/skills/design-interrogation/SKILL.md`: nine questions per mechanism, an adversarial second pass at milestones, three required artefacts; narrow triggers.
- `docs/references/2610.39380-learning-to-formulate.pdf` and `docs/references/README.md`: the Formulaic paper, frozen with hash and a section map.
- this file.

### Files changed

- `README.md`: index rows for 01, 02, 12 and the change log.
- `.agents/skills/README.md`: row for `design-interrogation`.

No code, spec or seed data was changed. Nothing is committed.

### Design requirements from the conversations

Status: **decided** = the user decided; **open** = waiting for the user; **default** = assistant's choice, stands until pulled.

| # | Requirement | Where | Status |
| --- | --- | --- | --- |
| R1 | The MIP is assembled mechanically from chain + hardware model; the agent does not write MIPs per case; keep an agent-written arm for comparison | D-T1 | default |
| R2 | Template `cost` becomes `effects` (transactions, storage, overlap); hardware coefficients live only in the hardware model | D-T2, 02 §3 | default; timing open (D-open-5) |
| R3 | The hardware model is declarative data (resources, walls, operation reservation tables, efficiency, interactions, composition); simulator, MIP and reports derive from it; four fidelity levels share one parameter table | D-T3, D-T5, 02 §4 | default; timing open (D-open-5) |
| R4 | MIP = steady-state (modulo-scheduling) projection; MIP ↔ simulator consistency check without a GPU | D-T4, 02 §5 | default |
| R5 | Fine-grained resource sharing as shared resources discovered by interference microbenchmarks; completeness as measured coverage | D-T14, D-T15 | default |
| R6 | Gap decomposition routes every next action; attribution order correctness → ε → δ → Δpre → L_rep → irreducible | D-T6, D-T11 | default |
| R7 | FormulationPlan as structured object; deterministic default plan and chain enumerator; agent plans scored against the default | D-T18, D-T19 | default |
| R8 | One-screen task header + versioned access discipline; pulls logged | D-T20 | default |
| R9 | Loops with bridge objects; design loop outermost; staged release S0–S4 | D-T21, D-T22 | decided in substance |
| R10 | Independent processes, no multi-agent; session policy of 10 §4.2 kept; calibration a branch of the knowledge loop | D-T24 | decided in substance |
| R11 | One tunable controls table (12); a frontend settings page later; a knob that prevents nothing is removed | D-T25 | decided |
| R12 | System parameters set by small experiments X1–X9; defaults say so until then | D-T28 | decided |
| R13 | Capability replaces HwFeature; Realisation split from Template; Finding unifies Regime and Negative; HwModel split into class and instance; levels numbered 1–7 | D-T7…D-T10 | default |
| R14 | Unprobed mechanisms are recorded (`probed_on: []`), filtered by capability, counted as "known, unavailable here" | D-T27 | decided in substance |
| R15 | Backends: TileLang primary, CUDA C++ + PTX escape hatch, CuTe DSL not pursued for GB10, Triton for controls, MLIR not in phase one; per-backend expressibility coverage recorded | D-T29 | decided |
| R16 | Learning loop automated: intake → decompose (T2) → admit with rejection loop → reproduce → ablate → cover; reference directory = T2 deliverable format; `knowledge/seed/` = seed only | 02 §8.6 | default; D-open-2, D-open-4 |
| R17 | Coverage of SOTA reported in four levels; system-level features counted separately; a SOTA kernel yields mainly a chain + regime, rarely a template; composition statistics are a derived view | 02 §8.1–8.2 | default |
| R18 | Cost tiers L0–L4 with admission rules; "performance core" reproduction within ε = 10 %; ablation on our own families; `coupled_with` as first-class knowledge | 02 §8.4 | open (D-open-3, D-open-6) |
| R19 | Blind reconstruction (O12) as the learning loop's acceptance experiment | 01 §6 | open (D-open-7) |
| R20 | `design-interrogation` run on spec interface changes, new loops/roles/bridges, control changes, needs-review decisions, end of design conversations, milestones; not on implementation within a frozen interface | skill | default |
| R21 | 01 stays short (~250 lines); 02 is a staging area merged into numbered documents by construction | D-T30 | decided |

### What construction should do next, in order

1. Read 01, then 02, then 12. Treat 02 as proposals until the open decisions are answered.
2. Merge 02 into the numbered documents once the user confirms each part: §1 → 00; §2 → 23 and 21 §1; §3 → 22 §2; §4–5 → a new `25-hardware-model.md` plus 22 §3.3 and 10 §4.4; §6 → 10 §2 and 11 §1; §7 → 50 §4; §8 → 21 and 24. Each merge is a `design-interrogation` trigger (spec-facing).
3. Do not start schema changes (R2, R3, R13) before D-open-5 is answered.
4. Prepare S0: the GEMM golden path with no agent (WP7 scope), reporting % of SOL and the cuBLAS comparison, with every record visible.
5. Prepare L1 on the seven references: measurement under our protocol and the SASS instruction census (deterministic jobs).
6. Build the admission rejection loop and the "inexpressible" category (R16) so the seven decompositions can be imported as `proposed` (after D-open-2).
7. Add a `set_by` note to each row of 12 naming its X-experiment (12 §6), and record in `docs/decisions.md` any D-T entry the user confirms.

### What the user still has to decide

D-open-1 … D-open-7 in 02 §12.
