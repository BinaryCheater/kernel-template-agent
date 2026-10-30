# WP10a delivery — hand decomposition of reference kernels

Branch `wp/WP10a`. All files are under `knowledge/seed/references/` (one directory per reference;
conventions in `knowledge/seed/references/README.md`).

## What was done

Seven first-batch references from docs/design/24 §4 were decomposed by hand with the method of
`src/kta/agents/skills/decompose-kernel/SKILL.md`, in the fixed order below. Each was frozen to an
upstream commit (`source.yaml`) and every transaction is cited as `path:line` at that commit
(`decomposition.md`, `chain.yaml`, `reconstruction.yaml`). Each directory holds `source.yaml`,
`problem.yaml` (the workload), `reference.yaml`, `chain.yaml`, `reconstruction.yaml` with
`unmatched_transactions`, one `gaps/ux-NN-*.yaml` per unmatched transaction, and the Templates
and Patterns that reference proposed (`templates/`, `patterns/`, status proposed, assurance
declared).

These are **analysis of source code**. Nothing was built, run or measured on GB10 or any GPU
(allowed by spec/workplan.md WP10a); `runnable_on: [gb10]` means ISA eligibility only (D104).
No store records exist, so the evidence manifest has no record IDs (WP10a is not in the
record-backed list of D84).

### Per-reference results

Match rate = chain steps matched by a Template that existed before the reference, without a Gap
(D102). Pool = assumed seed (5 Templates, 8 Patterns, D103) plus everything proposed by earlier
references.

| # | Reference (upstream @ commit) | Runs on GB10 | Steps | Matched | Match rate | New Templates | Unmatched tx = Gaps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `cutlass_ampere_tf32_gemm` (NVIDIA/cutlass @ e13f5cd4, example 14) | eligible | 18 | 6 | 33% | 9 | 13 |
| 2 | `flash_attention_2_fwd` (Dao-AILab/flash-attention @ a3a9dfa9, csrc/) | eligible | 25 | 19 | 76% | 5 | 8 |
| 3 | `flash_attention_3_fwd` (same commit, hopper/) | analysis-only | 29 | 20 | 69% | 8 | 12 |
| 4 | `marlin_int4_gemm` (IST-DASLab/marlin @ 1f25790b) | eligible | 23 | 17 | 74% | 5 | 9 |
| 5 | `flashinfer_batch_decode` (flashinfer-ai/flashinfer @ 732134f2) | eligible | 19 | 16 | 84% | 3 | 6 |
| 6 | `vllm_fused_moe` (vllm-project/vllm @ 2eae56c4, Triton + moe_align) | eligible | 12 | 12 | 100% | 0 | 4 |
| 7 | `vllm_rms_norm` (same commit, csrc/layernorm_kernels.cu) | eligible | 9 | 9 | 100% | 0 | 3 |
| | **total** | | **135** | **99** | **73%** | **30** | **55** |

Discovery curve (cumulative new Templates in decomposition order): 9, 14, 22, 27, 30, 30, 30.
Of the 35 Templates in the final pool, 12 were used by one reference only and 6 by exactly two
(incidence Chao2 = 35 + 12² / (2·6) = 47; with 7 references this is indicative only).

Template reuse across references (references using it / chain steps):
physical_storage 7/14, boundary_predication 7/13, output_tiling 7/11, reduction_partition 7/9,
operator_fusion 7/7, multi_buffer_rotation 6/9, compute_instruction_selection 6/7,
async_transfer 6/6, vectorized_access 5/5, shared_resource_binding, staged_relayout,
cross_lane_reduction, approximate_math_rewrite 4 each; the FlashAttention-3 and Marlin
mechanisms (warp specialization, ping-pong, stream-K partition, lock-ordered combine, prepack,
dequantization) were used once.

### Unmatched transactions by class (55 total)

| By level | | By pattern_guess | | By Gap.missing | | By attribution | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 operator | 7 | other | 13 | template | 37 | space | 35 |
| 2 tensor mapping | 15 | transfer | 10 | other | 7 | granularity | 10 |
| 3 instruction | 30 | reduce | 8 | family | 6 | intent_sass | 8 |
| 4 system | 2 | work_assign | 5 | pattern | 3 | verification | 2 |
| 5 microarch | 1 | store | 5 | relation | 1 | | |
| | | map 4, contract 3, issue_order 3, sync 2, access 1, scan 1 | | check | 1 | | |

What the 55 are (hand classification of the `gaps/` files):

| Kind | Count | Status |
| --- | --- | --- |
| Step that needed a Template proposed in the same reference | 32 | resolved by that proposal; the Gap records the miss |
| Partial match the pool still cannot express | 4 | open: register double buffering (cutlass, Marlin), FlashAttention-2 phase-offset single slot, FlashAttention-3 async-proxy fence before a TMA store |
| Transaction explained by no chain step | 19 | open: compiler-decided allocation / lowering 7 (D106), family-level dispatch 6, named-barrier ID binding 1, programmatic dependent launch 1, dequant/MMA interleave left to nvcc 1, atomic histogram 1, prefix scan 1, launch-dependent reduction order 1 |

### New Patterns (proposed, D105)

`work_assign` (assignment and order of work units; cutlass), `issue_order` (issue order within or
across executors; flash_attention_3), `scan` (prefix computation; vllm_fused_moe). All three use
category `other` because spec/schemas.md's fixed category list has no scheduling or scan entry.

### New Templates (proposed, assurance declared)

| Proposed in | Templates |
| --- | --- |
| cutlass_ampere_tf32_gemm | output_tiling, compute_instruction_selection, operand_precision_conversion, collective_fragment_load, vectorized_access, boundary_predication, operator_fusion, staged_relayout, block_rasterization |
| flash_attention_2_fwd | online_rescaled_reduction, masked_tile_pruning, accumulator_operand_reuse, approximate_math_rewrite, cross_lane_reduction |
| flash_attention_3_fwd | warp_specialization, register_rebalancing, per_slot_handshake, cross_iteration_software_pipelining, pingpong_schedule, persistent_tile_scheduler, cluster_multicast, cache_eviction_hint |
| marlin_int4_gemm | offline_operand_prepack, fused_dequantization, linearized_work_partition, ordered_global_combine, smem_partial_combine |
| flashinfer_batch_decode | indexed_gather, runtime_work_descriptors, split_combine_kernel |

Every Template has all §4.2 fields; obligations are listed with a planned method, none checked.

### How the YAML was checked

After merging WP1 (`integration` @ 6762cc0) all 123 YAML files (7 each of Source, Problem,
Reference, Chain, Reconstruction; 55 Gaps; 30 Templates; 3 Patterns) validate with
`kta.core.schemas.validate_body`; the gate validates the Reference, Chain, Reconstruction, Gap,
Template and Pattern documents itself. Before WP1 existed, the files were checked with a scratch
validator kept outside the repository (not part of this delivery), which also covers the checks
the models do not make: pydantic models transcribed from
spec/schemas.md §4.1–§4.5, §4.7 Gap and §4.8 Problem with `extra="forbid"`; expression free
variables (only declared variables, `hw.*`, `p.*`, `case.*`, allowed functions); chain bindings
cover exactly the Template's variables; `after` forms a DAG; pool order; `matched` recomputed;
`unmatched_transactions[i]` ↔ `gaps/ux-{i+1}` with equal level and pattern_guess. A second
script resolved all 741 cited line ranges against the frozen clones (all in range; 3 citations in
Pattern examples point to other references' repositories on purpose). The gate's own WP10a
classification (`tools/kta_gate/specific.py` on `integration`) was replayed on the tree: every
reference has Reference, Chain, Reconstruction and Gap count = unmatched count. In-range line
numbers do not prove the cited line says what the text claims; spot checks were done while
writing, not exhaustively.

### Import into the knowledge pool (after WP2 and WP4)

After merging `integration` @ c4081bb (WP2 feature list, WP4 `kta init --references`, D97 string
literals), WP4's import refused the batch with 16 errors. All were fixed in the data:

- 11 `unknown_hw_feature`: `requires_hw` now uses the names of
  `src/kta/targets/gb10/features.yaml` (D108). Six names were already WP2's; `tma` became
  `cp_async_bulk_tensor`; cluster_multicast now names `[cp_async_bulk_tensor, barrier_cluster]`.
  The multicast form of cp.async.bulk.tensor, createpolicy / `.L2::cache_hint` and lop3 have no
  feature entry and were dropped from `requires_hw`; each affected Template says so in
  `scope.text` and a YAML comment, and the three features are listed under Needed changes.
  fused_dequantization names `fma_f16x2` (HFMA2) instead of lop3.
- 1 `conflict`: FlashAttention-2 and -3 now carry one identical Problem body
  `attention_fwd_f16_d128` (D109), so both References share one Problem record.
- 4 `missing_required`: boundary_predication, staged_relayout and indexed_gather now `enable`
  vectorized_access instead of requiring it (D110). FlashAttention-3 predicates and re-lays out
  through TMA and vLLM fused_moe uses Triton masked loads, so neither chain lacks a step.

No chain item, binding, Gap or count changed; match rates are unchanged (99/135).

Real import check: the whole batch was imported with `kta.knowledge.import_seed` +
`import_references` into a scratch test store that also holds the 52 HwFeature bodies of WP2's
feature list (support unset). It imports with no error: 7 Source, 6 Problem, 7 Reference,
3 Pattern, 30 Template, 7 Chain, 7 Reconstruction, 55 Gap. The WP4 gate's own scratch store
holds no HwFeature records at all, so every non-empty `requires_hw` (10 names in 9 Templates)
still fails there as `unknown_hw_feature`. Emptying all `requires_hw` would remove the hardware
requirement information to satisfy that check, so it was not done; see Needed changes.

### Critique of the knowledge model

1. **Match rate is not coverage.** vllm_fused_moe and vllm_rms_norm match 100% of their steps yet
   have 4 and 3 transactions no step explains. The coverage page should show unmatched
   transactions next to the match rate, never the rate alone.
2. **Compiler-decided transactions have no home.** Register allocation (every CUDA reference) and,
   for Triton, the entire level-3 lowering are real transactions that decide whether chain steps
   hold (e.g. physical_storage in registers), but they are not design variables. A Chain needs a
   way to attach compiler output (registers/spills from Build, TTGIR or SASS) as evidence for a
   step instead of a Template.
3. **multi_buffer_rotation's single obligation P ≤ S − 2 + β is too coarse.** It rejects correct
   register double buffering (cutlass, Marlin) and cannot express FlashAttention-2's two barriers
   per iteration with staggered K/V refills; per-slot mbarriers needed a separate Template
   (per_slot_handshake). The paper's step 4 assumes one trailing barrier per iteration and
   asynchronous writers.
4. **Some steps refine several transactions at once.** warp_specialization changes the issuer of
   every load and MMA and introduces the role handshake; splitting it produces steps that are not
   programs on their own. A Template should be allowed to refine a set of transactions
   (`refines` is a single Pattern today).
5. **One `refines` Pattern is too narrow for output tiling and fusion.** Output tiling applies to
   the independent-output space of any operator (contract, row reductions, attention); it was
   filed under `map` with a precondition `p.independent_outputs`. Fusion likewise spans
   producer/consumer pairs of different Patterns.
6. **The Pattern category enum is closed but the needs are not.** Scheduling (work_assign,
   issue_order) and scan had to be filed under `other`; the enum would need `schedule` and `scan`.
7. **`p.*` has no vocabulary.** Templates use ~50 distinct `p.*` names in preconditions and costs
   (p.independent_outputs, p.rescalable, p.tile_bytes, ...). Nothing declares them, so the
   free-variable check cannot catch typos and Γ cannot be evaluated until a case defines them.
   Pattern params should list the semantic ones; size quantities belong to the workload.
8. **Dispatch is invisible to a Reconstruction.** 6 of 7 references choose among bindings by device
   or shape; this is Family-level (DispatchSpec), so a Reference needs one Chain per bucket or a
   dispatch record.
9. **Numeric obligations beyond error bounds are missing.** Batch invariance / determinism of a
   reduction (vllm_rms_norm), fp16 storage of cross-CTA partials (Marlin) and the dependence of
   split counts on the SM count (FlashInfer, Marlin) can only be written as numeric_risk prose.
10. **Granularity that worked well.** Small, mechanism-level Templates (boundary_predication,
    vectorized_access, cross_lane_reduction, smem_partial_combine with `combine_op =
    rescaled_merge`) were reused across operator families, which supports "know the mechanism,
    not the kernel name". The curve flattening after reference 5 is partly because references 6–7
    are simpler, not only because the pool saturated.

### Least certain judgements

1. The assumed seed variables (D103) decide three findings: register double buffering and the
   FA-2 slot are partial only because `multi_buffer_rotation` was assumed to have exactly
   (S, P, β); FlashInfer's split counts as matched only because `reduction_partition.num_groups`
   was assumed to allow runtime binding.
2. Counting Triton steps whose lowering is compiler-chosen (vllm_fused_moe mma, ab_async,
   ab_stages) as matched; their bindings are unverified until IR or SASS is attached.
3. Treating FlashAttention-3's warp specialization as one indivisible step rather than splitting
   it into role assignment and handshake.

## Fixtures and simplifications

- WP10a adds no code under `src/` and no `# kta:fixture` marker. The gate lists every marker in
  `src/`; the only one is inherited from WP1 via the `integration` merge:
  `src/kta/store/bench.py` (`# kta:fixture` on `_body`, synthetic bodies for the store benchmark,
  origin `fixture`, D156; declared in deliveries/WP1.md). WP10a does not use it.
- No reference was built, run, measured or reproduced on GB10 (spec/workplan.md allows this);
  `reproduction_error` is null everywhere; `runnable_on` is ISA eligibility (D104).
- Decomposed configurations are single configurations per reference (listed in each
  `decomposition.md`); compiled-out variants (dropout, FP8, paged/varlen paths, split-K serial,
  etc.) are excluded and named there.
- The five seed Templates are assumed, not read from WP4 output (D103).

## Known issues

- The Source body (D143) has no fields for the analysed file list, commit date or retrieval time;
  they are kept as text in `source.yaml` `note` (D107).
- Upstream sources are shallow clones in a session scratch directory; `source.yaml` has the URL
  and commit to re-fetch them.
- Line citations were verified to be in range, not semantically re-read one by one.
- The per-class counts above were tallied by a scratch script over `gaps/*.yaml`; the summary of
  "what the 55 are" is a hand classification.

## New decisions

- D100 seed reference file format (needs user review)
- D101 Source record fields (needs user review; replaced by D107 after WP1 defined Source in D143)
- D102 reconstruction counting rules (needs user review)
- D103 assumed seed Template variables (needs user review)
- D104 meaning of `runnable_on` in WP10a
- D105 three proposed Patterns (needs user review)
- D106 compiler-decided transactions recorded per CUDA reference
- D107 seed `source.yaml` uses the WP1 Source body (replaces D101)
- D108 `requires_hw` uses WP2's GB10 feature names; unlisted features dropped with a note
- D109 one shared Problem for FlashAttention-2 and -3
- D110 boundary_predication, staged_relayout and indexed_gather enable (not require)
  vectorized_access

## Needed changes

- **spec/schemas.md:** consider optional `paths` (analysed files) and `retrieved_at` fields on Source
  (D143 has neither; D107 keeps them in `note`); give `UnmatchedTx` an id or give `Gap` a
  ref to the transaction it records (today the one-to-one link is by file name and order, D102);
  consider `schedule` and `scan` in the Pattern category enum (D105); consider allowing a Template
  to refine several Patterns.
- **WP4 (seed):** the seed adopted the variable names of D103 (D206), so no chain binding needs
  renaming. Still open: scope or split `multi_buffer_rotation` so register buffers and
  per-operand barrier phases are expressible (its validator reports the three partial matches as
  `constraint_false` warnings).
- **WP4 gate / `tools/kta_gate` (reference import check, D215):** the scratch store used for
  `kta init --references` holds no HwFeature records, so any Template with a non-empty
  `requires_hw` is refused there even when the names are WP2's. Load the HwFeatures of the GB10
  feature list (`kta.targets.gb10.probe.load_features()`, or `Gb10Target.features()`) into the
  scratch store before importing references, or let `kta init` import them. With that, the batch
  imports with no error (checked as described under "Import into the knowledge pool").
- **WP2 (feature list):** three features the reference Templates need have no entry in
  `src/kta/targets/gb10/features.yaml` and were dropped from `requires_hw` (D108):
  `cp.async.bulk.tensor` with `.multicast::cluster` (cluster_multicast), `createpolicy` /
  `.L2::cache_hint` loads (cache_eviction_hint), and `lop3.b32` (fused_dequantization). `hw.*`
  limits used: smem_per_cta, max_threads_per_cta, regs_per_sm, named_barriers,
  max_cluster_size. Cost unit `sfu` is used besides dram, l2, smem, tensor, fma, issue, launch.
- **decompose-kernel skill:** its output directories `new_templates/` / `new_patterns/` differ from
  the names the gate classifies (`templates/`, `patterns/`, D87); align one of the two.
- **Knowledge model (for discussion):** a way to attach compiler outputs to chain steps, and
  dispatch records for References (critique points 2 and 8).

## Gate output

`KTA_DATA=~/kta-data scripts/gate WP10a` on `wp/WP10a` @ 0dd4f92, after merging `integration` @ c4081bb (WP2, WP4):

```
gate WP10a: PASSED (21 pass, 0 warn, 0 fail)
  PASS ruff format
  PASS ruff check
  PASS import layers
  PASS residue
  PASS test doubles
  PASS error handling
  PASS print
  PASS subprocess outside Executor
  PASS secrets
  PASS acceptance lock — 4 locked files unchanged
  PASS pyright
  PASS unit tests — 232 passed
  PASS contract tests — 2 passed
  PASS fake coverage — 1 fake(s) covered
  PASS registry (kta doctor, dev)
  PASS delivery note
  PASS fixture declarations — 1 declared
  PASS decision log — 166 decisions
  PASS skipped tests — none
  PASS evidence — 8 claim(s)
  PASS references — 7 references
```
