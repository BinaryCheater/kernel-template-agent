# WP10b delivery — automatic decomposition and coverage (code stage)

Branch `wp/WP10b`, cut from `integration` @ 004fc12. Code stage (D320, D323): no agent runs, no
spark builds, no new references. The loop was exercised on WP10a's seven references by
resubmitting their hand decompositions as T2/T3 deliveries through the human submission path.

## What was done

**Expand loop T2/T3 path** (`src/kta/orchestrator/expand_loop.py`, `expand_work.py`, expand part
of `machines.yaml`):

- `frozen` checks the freeze (as before). An analysis-only reference (empty `runnable_on`, or
  `kta expand --analysis-only REASON`, which writes a HumanAction) skips building and measuring;
  `measured` then records `reference_point: null` and creates T2 (D385). A runnable reference
  still blocks in `frozen` with NotSupported (needs a GPU worker; WP11).
- T2 inputs: Reference, Source, Problem; a re-match T2 also gets the previous Reconstruction.
- After T2 is accepted the loop reads its chain: every Template ref the chain names that the pool
  lacks gets one T3, with inputs Reconstruction, Reference, Source and the T2 Submission (whose
  `templates/` drafts the T3 starts from) (D382). Nothing to extract → `reconstructed`.
- `gap` waits until every T3 is accepted or cancelled. One accepted → a new T2 re-matches
  (round + 1, bounded by `max_rounds`); all cancelled → `reconstructed` with the Gaps open
  (D384). Escalated or suspended tasks stop the loop.
- `reconstructed` for an analysis-only reference → `covered` (new edge). Its work snapshots the
  pool (PoolSnapshot, `current_pool` by compare-and-set) and writes the coverage Report (D387).
  Reproduction and ablation of runnable references stay NotSupported.

**Done-checks** (`done_checks.py`, functions added; CHECKS/DONE_CHECKS extended):

- T2 reads the decompose-kernel skill's `out/` layout (`chain.yaml`, `reconstruction.yaml`,
  `gaps/*.yaml`, `patterns/*.yaml`, `templates/*.yaml` drafts) and still the single-file form
  (D381). `reconstruction_fields` uses the skill's counting rules (D380): `total` = chain items,
  `matched` ≤ `total`, at least `total − matched` unmatched transactions, unique keys, known
  `after`; proposed Patterns must not clash with the pool. `matched_templates_exist`: `matched` ≤
  chain items whose Template is in the pool; every item names `name@version`.
  `gaps_match_unmatched`: one Gap per unmatched transaction, in order, equal level and
  pattern_guess. WP6's previous rules rejected all seven WP10a decompositions.
- T3: `validate_template` against the store plus submitted Patterns plus the sibling T2 drafts
  (so FlashAttention-3's mutually related Templates can be accepted; D383;
  `propose_template(validate_in=)` added); new `template_fills_chain`: the Template is named by
  the input Reconstruction's chain and not yet in the pool.
- Writers (`outputs.py`): T2 writes Chain (with `refs.templates` for known Templates), Gaps,
  proposed Patterns and the Reconstruction; T3 writes submitted Patterns, then the Template.

**Coverage** (`src/kta/knowledge/coverage.py`, `types.py`; D386):

- Only the latest Reconstruction per Reference counts (the re-match writes one per round).
- Every item has a value: zero-filled levels, assurances, Gap `missing` and `attribution`;
  `families_per_template` lists every Template; new `gaps_by_missing`, `gaps_by_attribution`,
  `templates_missing`, `direct_test`, Chao1 95% interval, and `pending` naming the 21 §2
  quantities that need real runs (instruction coverage, gap explanation rate, input-space
  coverage).
- `coverage_page(...)` maps the report to the coverage page's fields (spec/modules/api-web.md
  "Coverage", `web/src/data/types.ts` `CoverageReport`); the stored Report has those fields plus
  `report` (the full CoverageReport) and the rng seed. WP9b's `/coverage` view serves it
  unchanged.

**Runs** (evidence: `deliveries/WP10b.evidence.json`):

- `tests/unit/test_expand_replay.py`: all seven references go `frozen` → `covered`; 12 T2 and
  30 T3 accepted (per reference 9/5/8/5/3/0/0 T3s); 37 Templates in the pool (7 seed + 30); the
  last coverage Report has 7 reconstructions, 99/135 matched, 55 unmatched transactions = 55
  Gaps (37 template), nothing left to extract, a value for every item. Also: cancelled T3s leave
  9 Templates open and lead to `covered`; an inflated `matched` is rejected.
- `tests/unit/test_expand_cli.py`: the same for cutlass through `kta init`, `kta reference add`,
  `kta expand --analysis-only`, `kta task submit --check`, `kta loop step`, `kta coverage`.
- Real dev-store run, kta commands only (`scripts/replay-references`, commit b65457d, store
  `~/kta-data/dev/WP10b-replay/`, output `evidence/replay-b65457d.txt`): all seven loops
  reached `covered`. Final coverage Report `259b8b28…`: 35 Templates observed, Chao1 47.0
  (95% 38.2–80.3; WP10a computed the same 47), Good–Turing 0.891, discovery 14, 19, 27, 32,
  35, 35, 35 (cumulative, seed Templates included), direct test 2 of 7 references with every
  step matched and 0 with nothing unexplained, 55 Gaps; all seven references shown `covered`
  on the page data. A first identical run from a tree with the script uncommitted is in
  `~/kta-data/dev/WP10b/` (src identical to 6b4cabe; records marked dirty).

## Fixtures and simplifications

- No fixture code. `tests/replayhelp.py` and `scripts/replay-references` are a scripted human:
  they resubmit WP10a's files unchanged; they produce no agent output.
- All seven references ran analysis-only by your flag (D385), although six have
  `runnable_on: [gb10]`: nothing was built, measured, reproduced or ablated.
- The scripted human submits sibling T3s in relation-dependency order (see Known issues).

## Known issues

- **Parallel T3s and relations.** A T3 whose Template names (in `relations`) a sibling Template
  that has no draft in the T2 Submission is rejected with `unknown_relation_target` until the
  sibling is accepted, using a retry. With drafts present (the skill's layout) it is accepted.
- **Gaps are not linked to Templates.** Only Gap text says "Proposed: X@1"; coverage cannot tell
  resolved from open Gaps, so `gaps_by_missing` counts every recorded Gap (37 "template" Gaps,
  of which WP10a says 32 are resolved by its proposals). D382 review item.
- **D102 `matched` in re-match rounds.** The re-match T2 resubmits WP10a's numbers (`matched`
  counts Templates that existed before the reference); the loop's transition to
  `reconstructed` uses "every chain Template is in the pool" instead, so `matched < total` can
  reach `covered`.
- **Seed versions.** The seed holds `output_tiling@2` and `compute_instruction_selection@2`
  (WP7); WP10a's chains name `@1`, so the loop extracts the `@1` versions as new Templates.
- **`kta reference add`** links `refs.problem` only when `reference.problem` is a record ID, so
  CLI-added references get T2 inputs without the Problem (the Python path links it).
- **Pool snapshots.** `covered` writes a snapshot of every knowledge record; a concurrent
  snapshot elsewhere blocks the loop with a ConfigError (re-run `kta expand`). No snapshot is
  taken after an accepted task (WP11).
- **Shared files.** To make the path work I changed WP6-owned code: `outputs.py` (T2/T3
  writers), `kta expand --analysis-only` in `cli/loop_cmds.py`, and expectations in
  `test_orchestrator_step.py`, `test_orchestrator_tasks.py`, `test_cli_orchestrator.py`;
  WP4's `propose_template` (`validate_in`) and `test_knowledge_use.py`.
- `tests/acceptance/WP3` errors locally (needs spark with this commit deployed); unrelated.

## New decisions

D380, D381, D382 (needs user review), D383, D384 (needs user review), D385 (needs user review),
D386, D387 (needs user review), D388. Review items for D382, D384, D385, D387 are in
`docs/review-queue.md`.

## Needed changes

- **WP9b (web):** the Coverage page may show the new `pending` map (why
  `instruction_families` and `gap_explained` are empty) and `report.direct_test` /
  `report.gaps_by_missing`; nothing is required for the page to render.
- **WP6 (cli):** `kta reference add` could accept a `problem` body and link it; `kta coverage`
  prints only three keys.
- **WP11:** if the learn process owns pool snapshots, `expand_work.write_coverage` should call
  its function (D387).
- **WP8 (skill decompose-kernel):** say that `templates/` drafts go to T3 tasks and `patterns/`
  are written on acceptance (D381).

## Pending real runs

- Add at least 3 new reference implementations (none added now, per D323) and take each from
  `frozen` to `covered` (or a Gap) with real agent T2/T3 runs in `dev/WP10b`; list the coverage
  Report in the evidence (full gate, D388).
- Build and measure runnable references on spark (`frozen`, `measured`), reproduce from a family
  and ablate (`reconstructed`, `reproduced`, `ablated`): NotSupported until WP11.
- Coverage items that need real runs: instruction coverage, gap explanation rate, input-space
  coverage (`pending` in the report).

## Gate output

```
gate WP10b --stage code: PASSED (23 pass, 1 warn, 0 fail)
  PASS ruff format
  PASS ruff check
  PASS import layers
  PASS residue
  PASS test doubles
  PASS error handling
  PASS print
  PASS subprocess outside Executor
  PASS secrets
  PASS acceptance lock — 7 locked files unchanged
  PASS pyright
  PASS unit tests — 491 passed
  PASS contract tests — 5 passed
  PASS fake coverage — 1 fake(s) covered
  PASS registry (kta doctor, dev)
  PASS delivery note
  PASS fixture declarations — 1 declared
  PASS decision log — 290 decisions
  PASS skipped tests — none
  PASS evidence — 2 claim(s)
  PASS WP10b tests — 11 tests passed
  PASS expand transitions — T1-T3 only; covered reached
  PASS T2/T3 done-checks — every named check exists
  WARN WP10b real expand loop — pending real run (code stage): at least 3 new references taken from frozen to covered (or a Gap) by agent T2/T3 runs in dev/WP10b; coverage Report in the evidence
```
