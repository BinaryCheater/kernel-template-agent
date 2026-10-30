# WP4 delivery: knowledge

Branch `wp/WP4`, cut from `integration` @ `6762cc0` (WP0, WP1, WP13 merged). Code commit of the evidence runs: `f0cb2d01c9c816733dba01439c8788752936950c` (clean). Dev store: `~/kta-data/dev/WP4/kta.db` on the control host.

## What was done

Real runs (`deliveries/WP4.evidence.json`, records in `~/kta-data/dev/WP4/kta.db`):

- **`kta init` imports the seed (claim 0).** `KTA_ENV=dev KTA_WP=WP4 kta init` created the WP4 dev store and imported 23 records in one batch: 1 Source (the paper), 8 Patterns, 9 Structures and the 5 GEMM-step Templates. All are origin `real`, written by clean commit `f0cb2d0`, actor `import`/`kta init`. Every Template was checked with `validate_template` against the store plus the batch before anything was written.
- **Import is idempotent (claim 1).** A second `kta init` wrote nothing and reused all 23 records.
- **Read-only check on the dev store (no new records).** On the same store:
  - `validate_template` returned no Issue for any of the five Templates.
  - `retrieve_templates(pattern="store")` returned the two store Templates as 4-line cards.
  - `coverage_report` gave assurance `{declared: 5}` and the HwFeature × Pattern grid.
  - `render_text_knowledge` wrote `index.md`, 8 Pattern pages and `structures.md`.

The knowledge module (`src/kta/knowledge/`), covered by unit and acceptance tests:

- **Patterns** (`patterns.py`): `load_patterns` (usable Patterns from the store) and `match_pattern`, a deterministic score (D208). When nothing scores, it returns `[("other", 0)]`.
- **Templates** (`templates.py`): `validate_template` checks:
  - non-empty fields, Σ and X;
  - expression parsing with free variables (C/Pre/cost/`when` over X; Γ over `p.*`, `hw.*` and `case.*` only);
  - `hw.*` names against a given hardware contract (warning);
  - domains, unique names and IDs, `same_as` target and verbatim equality (K1), level order;
  - that refines/produces Patterns, `requires_hw` HwFeatures and relation targets exist;
  - duplicates, reported as a `possible_duplicate` warning (D201).

  `propose_template` writes a Template only when valid. It refuses a changed body under an existing `name@version` and links `previous` (D202, D203).
- **Chains** (`chains.py`): `validate_chain` checks:
  - Templates exist;
  - bindings cover every variable, with no unknown variable;
  - constant bindings lie in their domains;
  - `p.<param>` bindings are allowed (D205);
  - after-keys form a DAG;
  - chain-wide `requires`/`excludes` relations.

  It also evaluates constraints whose variables are all bound to constants; a false one gives a warning (D214).
- **Families** (`families.py`): `load_family` reads the directory layout of spec/schemas.md §4.4. `validate_family` checks:
  - the chain;
  - that bindings name family variables or domain values;
  - that the family variables equal the bound variables ∪ glue ∪ dispatch;
  - that glue constraints have a meaning and parse;
  - `kta:step <marker> begin/end` in the generated source for the default assignment;
  - that `generate()` returns for every edge shape (D213).
- **Scope and reuse** (`scope.py`): `scope_holds` evaluates Γ once per shape of the case and returns true, false or unknown (D209). `reuse_plan` gives keep/invalid/unknown per chain step, with `alternative_of` candidates for invalid steps.
- **Retrieval** (`retrieve.py`): `retrieve_templates`, `retrieve_families`, `nearest_chains`, `diff_templates` and `card` (≤ 5 lines, author summary labelled). They apply hard filters, then rank by assurance and reuse count; steps whose Γ is unknown come last and are marked (D210).
- **Rendering and coverage** (`render.py`, `coverage.py`): `family_overview`, `render_text_knowledge` and `coverage_report` (D211). The report covers every quantity in knowledge.md:
  - discovery curve and 50-shuffle band, Chao1, Good–Turing;
  - per-reference match rate, unmatched transactions by level and pattern;
  - count of `other` Patterns, HwFeature × Pattern grid, assurance distribution, families per Template.
- **Seed import** (`seed.py`, `pending.py`): `import_seed` and `import_references` validate a whole batch through a read-only `PendingView`, then write it with one `put_many`, so a refused batch writes nothing. Records are reused by identity key, so import is idempotent, and a changed body under an existing key is a conflict (D204). References are imported in the pool order given in their README.
- **Seed** (`knowledge/seed/`):
  - `sources/`: the paper, at its upstream commit.
  - `patterns/`: reduce, contract, map, store, transfer, sync, access and other, each with parameters and at least two examples.
  - `structures/`: the nine exploitable structures (objective / variables / enumeration).
  - `templates/`: `reduction_partition`, `physical_storage`, `async_transfer`, `multi_buffer_rotation` and `shared_resource_binding` (D206). Step 5's chip variables are fixed through `hw.*`, and obligations list their planned methods.
- **CLI**: `kta init [--references] [--json]` (D212); config `[knowledge] seed_dir`.
- **Gate**: `wp4` in `tools/kta_gate/specific.py` (D215): the seed imports and validates in a scratch store, the five Templates, 8 Patterns and 9 Structures exist, and the named WP4 acceptance tests passed. `seed/references` is reported as WARN when it fails. Unit-tested in `tests/unit/test_gate_wp4.py`.
- **Tests**:
  - `tests/acceptance/WP4/test_knowledge_acceptance.py` (7 tests): the five seed Templates validate; the eight Patterns are present; error classes for Templates, Chains and Families, one counterexample each; free-variable checks; duplicate hint.
  - `tests/unit/test_knowledge_pool.py` and `tests/unit/test_knowledge_use.py` (20 tests).

## Fixtures and simplifications

- WP4 adds no `# kta:fixture` to `src/`. Inherited from WP1, unchanged: `src/kta/store/bench.py` (`# kta:fixture` on `_body`), the synthetic record bodies of the store benchmark (D156). The test family under `tests/knowhelp.py` (`family_dir`) is test data for the validator tests, not a kernel.
- Simplification allowed by WP4: retrieval ranking does not use Solutions (%-of-bound), which do not exist yet. It ranks by assurance and reuse count (D210).
- The seed Templates' obligations are listed only (assurance `declared`); none is checked. This is what knowledge.md asks for.
- No skipped tests.

## Known issues

- **No string comparisons in expressions (D216).** Conditions on string-valued variables cannot be written as C/Pre/Γ, for example "mechanism = instruction needs cp.async" or "level = smem bounds capacity by `hw.smem_per_cta`". The seed states them in obligations and Γ text, and `requires_hw` is empty for all five Templates (D207).
- **`p.*` needs WP5 (D205).** How a family binds `p.size_bytes`, `p.capacity_bytes` and the other `p.*` parameters to expressions over family variables is left to WP5's formulation assembly. Chains may bind them to constants today.
- **No knowledge-relation edges.** Relations (refines, requires, alternative_of) are resolved by name through `Store.query`, not stored as graph edges. Retrieval scans the pool, which is fine at thousands of records (docs/design/23).
- **`validate_family` runs `gen.py` in-process (D213).** It is not yet exercised on a real family: WP7 builds the first one.
- **WP10a's reference decompositions do not import yet.** WP10a is not yet merged into this branch. I validated its current tree read-only from its worktree, and the batch is refused; details under "Needed changes". The WP4 gate reports this as WARN (D215).
- `match_pattern` is a heuristic for ranking candidates, not a classifier.

## New decisions

D200, D201, D202, D203, D204, D205 (needs user review), D206 (needs user review), D207, D208, D209, D210, D211, D212, D213 (needs user review), D214, D215 (needs user review), D216 (needs user review).

## Needed changes

- **core (WP1 owner, locked for WP4): string literals in expressions (D216).** Allow string literals in `==`/`!=` (and optionally `in` over a literal list) in `kta.core.expr`, treated as non-linear. 13 expressions in WP10a's Templates use them.
- **WP10a: its reference tree, validated read-only (worktree `wp-wp10a-679ff2` @ `a307bd0`).** All files parse and every body validates against the WP1 models. The batch is refused for these reasons:
  1. String literals in C/Pre/`expected_sass.when`: boundary_predication, collective_fragment_load, compute_instruction_selection, operator_fusion, staged_relayout, accumulator_operand_reuse, approximate_math_rewrite, per_slot_handshake, persistent_tile_scheduler, fused_dequantization and split_combine_kernel. Fixed by the core change above, or by numeric encodings.
  2. `requires_hw` names with no HwFeature record: ldmatrix, shfl_sync, l2_cache_hint, tma, cluster_multicast, wgmma, mbarrier, named_barrier, setmaxnreg, lop3. These must come from WP2's `targets/gb10/features.yaml` with exactly these names, or the Templates must use WP2's names.
  3. Problem `attention_fwd_f16_d128` is defined twice with different bodies (flash_attention_2_fwd and flash_attention_3_fwd). Give one of them another name, or make the two bodies identical.
  4. `requires` relations unmet in a chain: in flash_attention_3_fwd, boundary_predication and staged_relayout require vectorized_access; in vllm_fused_moe, boundary_predication and indexed_gather require vectorized_access, which the chain does not contain. Either the relation is `enables`, or the chain lacks a step.

  WP10a's chains bind the five seed Templates with the variable names they assumed (D103). No renaming is needed. Expected warning once importable: the register double buffer in cutlass_ampere_tf32_gemm (`frag_stages`: S=2, P=1, beta=0) violates `P <= S - 2 + beta`. WP10a already marks this as a partial match (gap ux-09).
- **WP2:** HwFeature names as listed above. `validate_template` errors on `requires_hw` names that are not HwFeature records.
- **WP5:** the meaning of `p.*` bindings in families (D205).
- **Spec edits made in this WP:** spec/modules/knowledge.md signatures take the store explicitly (D200), plus the seed functions and files.

## Gate output

Run: `KTA_DATA=~/kta-data scripts/gate WP4` on the control host, after the evidence runs (WP10a not merged yet, so no reference WARN line).

```text
gate WP4: PASSED (22 pass, 0 warn, 0 fail)
  PASS ruff format
  PASS ruff check
  PASS import layers
  PASS residue
  PASS test doubles
  PASS error handling
  PASS print
  PASS subprocess outside Executor
  PASS secrets
  PASS acceptance lock — 2 locked files unchanged
  PASS pyright
  PASS unit tests — 190 passed
  PASS contract tests — 2 passed
  PASS fake coverage — 1 fake(s) covered
  PASS registry (kta doctor, dev)
  PASS delivery note
  PASS fixture declarations — 1 declared
  PASS decision log — 133 decisions
  PASS skipped tests — none
  PASS evidence — 2 claim(s)
  PASS WP4 acceptance tests — 7 tests passed
  PASS seed validates — 8 Pattern, 1 Source, 9 Structure, 5 Template
```
