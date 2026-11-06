# Template replay state

The [research brief](research-brief.md) connects these results to the proposed
precompile L2 and MIP work. Start a new implementation session with the
[Claude handoff prompt](claude-handoff.md).

Latest: [E3](results/template-chain-e3.md) ran agents through form, MIP and GPU on two production comparators, with no visible Template effect in two repeats per arm. Before it, the [decision-form pilot](results/decision-form-pilot.md) searched three kernels with a real MIP. Findability is measured on branch `claude/template-recall-e2-c054b3`, not merged here; its tools are copied onto this branch.

The exact-source attention campaign has finished all nine model sessions and all scheduled final outcomes. Nine selected implementations passed hidden O/LSE correctness checks. Seven have valid three-window timings after the common frozen-source measurement recovery. The fixed performance target was attained by one self-selection run and one no-Template run; no prescribed run attained it. Two measurements remain unresolved.

Read [final findings](controlled-replay/conclusion.md) first. It includes all nine outcomes, original-to-generated microseconds, per-session active/GPU-wait time, tokens, retained-price estimates, and execution problems. [Initial confirmation](controlled-replay/results.md) remains separate from the fixed-source follow-up.

- Comparator: the pinned, unchanged FlashAttention-2 CUDA causal FP16 D128 forward source kernel, with a thin host binding, on GB10. Inputs are BSHD [1,S,8,128], S=512/2048/513. Both O and natural-log LSE are required.
- Information conditions: prescribed 18 rules; selection from 312 paired canonical entries plus eight unpaired proposals; no Templates. Each has three independent native sessions. All nine ran concurrently; GPU work was serialized.
- Execution method: agent-authored TileLang with agent-chosen concrete parameters and iterative GPU timing feedback. No MIP solver or KTA performance-prediction model was invoked; the Template-to-solver/performance-model search pipeline was not evaluated.
- Finding: Template-assisted reconstruction and direct coding both reproduce the computation and can reach the performance target. The pilot does not establish an efficiency advantage from Template information. A second self-selection candidate repeatedly ran faster but missed the full stability protocol; its numbers and source are retained.
- Execution problems: all sessions had a common early infrastructure interruption; run-03 was prematurely closed by a coordinator receipt mismatch with budget remaining; run-08 read its own raw development-feedback records containing development seeds. Model contexts and original remaining budgets were retained. No model was reopened or kernel repaired after hidden validation.
- Frozen-source follow-up: all four initially unresolved timings received exactly three additional windows, within each source's remaining original 300-second confirmation budget. Initial outcomes were retained. No additional model calls.
- Accounting: 81.4 minutes from the first model session to the final measurement, including interruption and follow-up, excluding prior preparation. Nine-session model usage reconciles with native per-request records. API-equivalent estimate is $11.373; actual billing is not reported.
- Evidence: /Users/dingyi/kta-data/dev/source-attention-control-v1/ contains immutable inputs, all native sessions, source versions, failures, paired samples, audits, initial confirmation, frozen-validation-v1/, final-findings.json, final-accounting.json and campaign-completion.json.
- Earlier generic experiments remain under /Users/dingyi/kta-data/dev/template-replay/ and results/scope-correction.md. They do not substitute for the exact-source comparison.
- No further model or GPU experiments are running. The research checkpoint on `wp/LEARNDEMO1` contains the reports, Template library, replay tooling, plotting scripts, and handoff. Raw records and generated figures are stored at the evidence paths above.
- Figures: controlled-replay/figures.md indexes five scripted figure sets (performance, sample spread, raw iteration history, best improvement versus active time, time/cost), with PNG/PDF/SVG and CSV exports under /Users/dingyi/kta-data/dev/source-attention-control-v1/plots/v1/. No new measurements or model calls.
