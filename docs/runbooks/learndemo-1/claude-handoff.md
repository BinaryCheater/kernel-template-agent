# Claude handoff prompt

Copy the prompt below into Claude Code with the KTA repository open. The
experiment records are local to the original machine. A separate checkout on
that machine can read them at the paths below.

---

Take over KTA's Template-driven kernel prediction and search research from
branch `wp/LEARNDEMO1` in
`/Users/dingyi/Projects/agentic-codesign/kta`. Follow `AGENTS.md` and
`spec/conventions.md`. Work from the checkpoint on that branch and preserve
the existing experiments and L3 development.

Read these in order:

1. `docs/runbooks/learndemo-1/research-brief.md` for the research question,
   known results, and what a successful next experiment must show.
2. `docs/runbooks/learndemo-1/controlled-replay/conclusion.md` for the exact
   production comparator, correctness, timings, and limitations.
3. `docs/runbooks/learndemo-1/model-search-pilot.md` for the proposed design.
   Check current `docs/design/perfmodel/STATE.md` and
   `spec/modules/formulation-solve-models.md` before choosing what to reuse.

Start by implementing and running the smallest informative offline pilot.
Use one retained correct TileLang attention implementation as the starting
family. The completed self-selection run-06 is a candidate. Verify the
retained source, compile path, comparator, and artifacts first. Select a
small number of meaningful choices, for example tile size and pipeline
depth, and freeze the allowed domain. If you choose a different retained
implementation, record the reason and preserve the same original production
comparator and O/LSE contract.

Make one set of parameter bindings drive both deterministic source/launch
generation and a precompile execution description. Use L3's mechanism
knowledge and a few targeted compiler witnesses to verify the mapping.
Start with the current L3 and improve a mechanism only when pilot evidence
shows that it materially changes a search decision.
Choose compile or timing samples that distinguish hypotheses about the
mapping or performance. Keep hardware calibration and family-specific
correction terms separate. Generate and solve an actual MIP from the
supported symbolic costs and constraints. Verify the translation on a
small domain, then check whether its shortlist and L2 ranking find fast
correct kernels on configurations or shapes held out of calibration.
Save predictions before GPU evaluation. Compare with same-family empirical
tuning and report both selection quality and total effort, including
one-time preparation. Use the exact pinned FlashAttention-2 source kernel
for the original-versus-generated timings.

Use a small script and the existing working measurement path. Temporary
remote compilation and GPU measurement are authorized for the research
pilot. Follow the relevant remote and measurement skills and serialize
GPU timing. Keep work independent of product UI and runtime integration.
Retain all evaluated sources, compiler metadata, parameter assignments,
predictions, failures, raw timing samples, and native model sessions.
Report time, tokens, estimated cost where applicable, and GPU queue time
separately. Distinguish an actual MIP run from any enumeration control.
Record a compact protocol before evaluation, including sample selection,
calibration and holdout split, budgets, success criteria, and comparison
method. Make progress through to the first real result rather than
stopping with a new architecture proposal.

Evidence and code:

- `/Users/dingyi/kta-data/dev/source-attention-control-v1/` holds all nine
  sessions, generated source versions, original build and measurements,
  confirmation and frozen follow-up records, and final accounting.
  Start with `campaign-completion.json`, `final-findings.json`, and
  `final-accounting.json`. `plots/v1/` holds the derived figures.
- `/Users/dingyi/kta-data/dev/template-replay/` holds earlier exploratory
  replays. Keep their results separate from the exact-source campaign.
- `knowledge/library/` has the consolidated Template library.
  `tools/kta_source_replay/` and `scripts/source-replay` provide the
  controlled replay tooling. The earlier campaign used agent coding and
  empirical tuning. It provides no MIP or L2 validation result.
- The checkpoint contains research code and reports. Native transcripts,
  binaries, raw runs, and figure outputs stay in the data directory.
  Report any missing evidence instead of assuming it is in Git.

Deliver runnable code and retained experiment records, plus a short report
with original and generated microseconds, correctness, prediction versus
measurement, missed candidates, and the complete cost comparison. Report
what still limits prediction and which measurement would distinguish the
leading explanations. Expand to a studied quantization, fusion, or
irregular kernel after the first family works and the evidence supports
expansion. Keep that later transfer claim separate.
