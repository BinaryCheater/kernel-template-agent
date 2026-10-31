# knowledge

设计依据：docs/design/21、22、23、24。schema 见 spec/schemas.md §4。

## 文件

```
knowledge/types.py         Issue、Card、ScopeResult、ReusePlan、CoverageReport 等结果类型
knowledge/pool.py          经 Store 读取知识对象（按名字、状态）
knowledge/pending.py       Store＋待写入记录的只读视图（批量校验用）
knowledge/seed.py          seed 导入（kta init）
knowledge/patterns.py      Pattern 表的加载、新增、匹配
knowledge/templates.py     Template 的加载、校验、查重、版本
knowledge/chains.py        Chain 的校验；链项之间的关系检查
knowledge/families.py      族目录的加载与一致性检查
knowledge/scope.py         Γ 求值；增量复用的逐步检查
knowledge/retrieve.py      检索：Template、族、链、按诊断、差集
knowledge/render.py        卡片、族概览、文字知识渲染
knowledge/coverage.py      覆盖统计（21 §2）
knowledge/capabilities.py  Capability support per Target; Template availability (D442, D445)
knowledge/realisations.py  Realisations of a Template, legacy ones from schema-1 bodies; SASS expectations (D443)
knowledge/findings.py      Findings (incl. legacy Regime/Negative) and Couplings (D444, D447)
knowledge/ladder.py        the one ladder of empirical knowledge and its evidence rules (D446)
knowledge/promotion.py     promote, demote (with stale propagation), promotion review (D448)
knowledge/pooldiff.py      diff of two PoolSnapshots (D449)
knowledge/seed/            初始 Pattern 表、Structure 表、论文 GEMM 五步的 Template（YAML）；Capabilities, GB10 declarations, Realisations (WP14)
```

## 接口

```python
# Pattern
def load_patterns(store) -> dict[str, Pattern]
def match_pattern(tx: TxDescription, patterns) -> list[tuple[str, float]]   # 候选模式及匹配度；无匹配返回 [("other", 0)]

# Template
def validate_template(t: Template, store, hw: dict | None) -> list[Issue]   # Issue: kind, message, where, severity error|warning (D201)
    # 字段完整；表达式可解析、自由变量已声明；preconditions 的 same_as 指向存在的约束；
    # relations 的 target 存在；refines 和 produces 的 Pattern 存在；requires 的 Capability 存在
    # (D442: unknown name -> Issue "unknown_hw_feature"; a legacy HwFeature name -> warning "hw_feature_not_capability")；
    # 查重：同一 refines、同类 object.type、变量名集合与成本单元集合相同 → Issue(kind="possible_duplicate")
def propose_template(ctx, store, t: Template, sources: list[str]) -> str     # 校验通过后写入，状态 proposed，保证等级 declared（D202、D203）

# Chain 与 Family
def validate_chain(ch: Chain, store) -> list[Issue]
    # 每个 item 的 template 存在；bindings 覆盖该 Template 的全部变量；after 构成 DAG；
    # 链中 Template 的 requires/excludes 关系满足
def load_family(path: Path) -> FamilyPackage
def validate_family(fp: FamilyPackage, store, hw: dict | None) -> list[Issue]
    # validate_chain；每个 item 的 code_markers 都在 gen.py 生成的源码中出现（对一组默认取值生成后检查）；
    # 族变量 = 各 item 绑定到的变量 ∪ glue 中的变量，没有多余也没有缺失；
    # glue_constraints 每条有 meaning；generate() 对边界形状能正常返回

# 适用范围与增量复用
def scope_holds(t: Template, case: Case, hw: dict, params: dict) -> ScopeResult    # holds: bool | None（无法求值）、reason
def reuse_plan(store, chain_id: str, case: Case, hw: dict, params: dict | None = None) -> ReusePlan
    # 对链中每一步求值 Γ；返回 keep / invalid / unknown 三类及原因；
    # 对 invalid 的步骤，给出 alternative_of 的候选

# 检索
def retrieve_families(store, case: Case, k: int = 5, hw: dict | None = None) -> list[Card]
def retrieve_templates(store, pattern: str | None, unit: str | None, text: str | None, case: Case | None, k: int = 10, hw: dict | None = None, params: dict | None = None, target: str | None = None) -> list[Card]
    # target (default case.target): only Templates whose every required Capability is "yes" there (D445)
def unavailable_templates(store, target: str) -> dict[str, list[str]]   # {"unavailable": [...] (a capability "no"), "unconfirmed": [...] ("unknown")}
def nearest_chains(store, case: Case, k: int = 3) -> list[Card]
def diff_templates(store, reference_chain: str, current_chain: str) -> list[Card]     # 差集检索
def card(store, record_id: str) -> Card          # 五行以内：名称、作用、强弱项（来自证据）、状态、保证等级

# 渲染与覆盖
def family_overview(store, pattern_or_problem: str) -> str     # Markdown，一页
def render_text_knowledge(store, snapshot_id: str | None, out_dir: Path) -> None
def coverage_report(store, snapshot_id: str | None, rng: Random, targets: list[str] | None = None) -> CoverageReport   # 21 §2 的全部量
    # + availability: target -> available / known_unavailable_here / unconfirmed_here (D445); targets default: every known Target name

# Seed (kta init)
def import_seed(ctx, store, root: Path, code: CodeState | None = None) -> ImportResult
def import_references(ctx, store, references: Path, code: CodeState | None = None) -> ImportResult
```

# Knowledge model v2 (WP14)
def target_capabilities(store, target: str, names: Iterable[str] | None = None) -> dict[str, CapabilityState]
    # status yes|no|unknown and basis: newest CapabilityDecl, else inferred from realised_by HwFeature probes, else unknown;
    # a legacy HwFeature name is answered by its support table (basis legacy_hw_feature)
def template_availability(store, requires: Iterable[str], target: str | None) -> Availability   # available, missing, unconfirmed
def realisations_of(store, template_ref: str, backend: str | None = None) -> list[RealisationRef]   # incl. the legacy one
def expected_sass(store, template_refs: Iterable[str], backend: str) -> dict[str, list[SassExpect]]
def probed_on(store, template_ref: str) -> list[str]           # Template's and its realisations' probed_on
def validate_realisation(r: Realisation, store) -> list[Issue]
def findings(store, subject=None, kind=None, target=None, usable_only=True) -> list[FindingRef]   # status "prior" out of scope
def couplings(store, member: str | None = None, target: str | None = None) -> list[CouplingRef]
def rule_problems(store, subject: Record, to_status: str, evidence: Iterable[str], controls: Controls) -> list[str]   # ladder (D446)
def promote(store, subject: str, to_status: str, evidence: list[str], now, controls: Controls) -> str   # checker; PermissionDenied when a rule fails
def demote(store, subject: str, counter_evidence: list[str], reason: str, now) -> Demotion   # system; one step; stale propagation
def review(store, now, controls: Controls) -> str             # Report "promotion_review" with candidates; changes nothing else
def diff(store, a: str, b: str) -> PoolDiff                    # pooldiff: added, removed, changed, status moves, counts

(D200) Functions that read the pool take the `Store` explicitly (no global state, conventions §3.1); `snapshot_id=None` means the whole store; `rng` is `ctx.rng("coverage")`. The package registers no interface implementation. Seed layout and atomic import: D204; `p.*` in Template expressions and chain bindings: D205.

## 行为

- **开放的表。** Pattern、HwFeature、Structure 都存在存储中，可以随时新增。代码中不得写死 Pattern 的枚举；`category` 的取值是 schema 中固定的少数几种，含 `other`。
- **查重是提示，不是拒绝。** `possible_duplicate` 会写入 Issue 和 Submission 的检查结果；worker 必须在交付说明中解释区别，否则完成检查不通过。
- **检索排序**：先硬过滤（Pattern 匹配、Γ 成立或无法求值、requires 的每个 Capability 在当前 Target 上为 yes，D445），再按以下顺序排序：保证等级、在相似 case 上的 %下界（来自 Solution）、复用次数。Γ 无法求值的排在后面，并在卡片上标明。
- **卡片和概览只从证据生成**，不读取作者写的评价性文字（summary 除外，且标明是作者说明）。
- **覆盖统计**（`CoverageReport`）至少包括：
  - Template 的发现曲线（按 Reconstruction 的时间顺序，随机重排 50 次）、Chao1 估计、Good–Turing 覆盖率；
  - 各 Reference 的匹配率、未匹配事务数（按 level 和 pattern_guess 统计）；
  - Pattern 中 `other` 的数量；
  - Capability × Pattern 网格（`hw_pattern_grid`，键 `none` 表示不需要能力）；
  - 每个 Target 上可用、"known, unavailable here"、"unconfirmed here" 的 Template 数（D445）；
  - 保证等级分布；
  - 每个 Template 出现在多少个族中。

## 初始知识（seed）

`knowledge/seed/` 中放：

- Pattern 表：reduce、contract、map、store、transfer、sync、access、other，各带参数和两个例子；
- Structure 表：论文表 3 的九项；
- 论文 GEMM 五步对应的 Template：reduction_partition、physical_storage、async_transfer、multi_buffer_rotation、shared_resource_binding（第五步在 GB10 上的芯片变量取固定值）。字段按 schemas.md 写全，`obligations` 只列出，`method` 写计划的检查方式。

## 验收

- 单元测试：各校验函数对正确输入无 Issue，对每类错误各有一个反例；表达式自由变量检查；查重。
- 真实运行：WP7 的 GEMM 族通过 `validate_family`；对它的链做 `reuse_plan`，换一个 dtype 或形状时，Γ 不成立的步骤被正确标出。

## WP7 implementation notes (D280, D286)

- `reuse_plan(store, chain_id, case, hw, params=None, assignment=None)`: with an assignment of the family variables, `p.*` expression bindings are evaluated per shape before Γ (`scope_holds(t, case, hw, params, bound=None, assignment=None)`). `problem_params` accepts a Problem record ID.
- The seed holds seven Templates: the paper's five GEMM steps plus `output_tiling@2` and `compute_instruction_selection@2` (D280); `knowledge/seed/families/gemm_fp16_tilelang/` is the first family and `knowledge/seed/cases/` holds its Case files (Problem, Contract, Case, baseline).

## WP14 knowledge model v2 (D440–D454)

- **Objects.** Template requires abstract Capabilities; Realisations carry code fragments, ISA constraints and expected SASS per backend × ISA range; Targets declare Capabilities by CapabilityDecl records; Findings replace Regime and Negative (legacy records are read as Findings); Couplings record `coupled_with` with evidence and scope. Levels are 1–7 (D441).
- **Unprobed mechanisms.** A Template whose required Capability no target provides stays in the pool (`probed_on: []`); retrieval on a target hides it; coverage counts it per target.
- **Ladder and engine.** One status ladder for empirical knowledge with evidence rules (D446); `promote` (checker), `demote` (system, one step, then `kta.store.stale.propagate`), `review` proposes promotions into the hardware model, a Template's Γ or a Structure's regime, and new Patterns. Thresholds come from the controls registry (D455): `escalation.finding_ladder.replicate_min_independent` (2) and `knowledge.promotion_review` (`min_targets_for_scope` 2, `min_repeats_for_pattern` 3); callers pass `ctx.controls`, and the Reports record the controls version.
- **Seed layout additions.** `capabilities/*.yaml` (Capability), `targets/<target>/*.yaml` (CapabilityDecl), `realisations/*.yaml` and `references/<ref>/realisations/*.yaml` (Realisation, refs.template resolved from `template: name@version`). Seed migrations are scripts under `scripts/migrations/` recorded in `knowledge/seed/migrations.yaml` (D450).
- **SASS checks** take expectations from realisations on the family's backend (`orchestrator.pipeline.chain_sass`, D453).
## WP15 implementation notes (D460, D465, D468)

- Templates declare `effects` (spec/schemas.md §4.2.1) instead of `cost`; a schema-1 `cost` body upcasts on read (D461). `validate_template` checks every effects expression like a constraint, refuses a storage level naming a non-level variable (`bad_storage_level`), and warns `unpriced_operation_class` for a literal transaction kind that no HwArch in the store prices (kinds chosen by a variable are counted by L1 instead). Duplicate detection compares transaction kinds instead of cost units.
- `import_seed` also imports `knowledge/seed/hardware/`: `arch_*.yaml` as HwArch, `model_*.yaml` as HwModel with `refs.arch` (identity `(name, version)` and `(target, version)`); an instance naming an unknown class is an `unknown_arch` issue.
- `retrieve_templates(unit=...)` matches a resource through the signature, legacy cost units, or the reservation tables (of every stored HwArch) of the Template's transaction kinds.
- Seed migration: `scripts/migrations/WP15_effects.py` (idempotent, `--check`) rewrote the `cost:` block of all 37 seed Templates in place, keeping versions (D468). Pattern `sync` gained the parameter `warps`.


## WP20 learning-loop runner (D560–D569)

Data: `src/kta/knowledge/learning.yaml` (version 1), read by `kta.knowledge.learning`. Every table is open; an unlisted name counts as `other`.

| Table | Content | Reader |
| --- | --- | --- |
| `admission` | Issue kind (+ optional message regex) → `error` / `warning` / `language_gap` | `issue_class`, `classify` |
| `system_features` | feature → regexes over unmatched-transaction descriptions; `pattern_guess: system` also counts | `system_feature` |
| `tiers` | L0–L4 with work, cost, yield and the admission rule name | `tiers(ReferenceFacts)`, `TIER_RULES` |
| `source_classes` | open_code, closed_library, paper, vendor_doc, other → pipeline stages | `source_class`, `stages` |
| `priority` | boost per reason | `priority(ReferencePriority)` = cases × gap closable × uncertainty × Π(1 + boost) |
| `performance_core` | what is included (ε is the knob `learning.reproduction_epsilon`), host inputs fixed, system features excluded | `performance_core(...)` → `PerformanceCore` |
| `ablation` | suspect kinds, repeats (run limits and noise band are knobs `learning.ablation_pairwise_max_runs`, `learning.ablation_max_runs`, `learning.coupling_noise_band`) | `kta.knowledge.ablation` |
| `sota_levels`, `backends`, `o12` | the four coverage levels, backends counted for expressibility, the O12 protocol (defined, not run) | `kta.knowledge.sota`, `o12_protocol` |

**Issue severities.** `Issue.severity` is `error`, `warning` or `gap`. `validate_template` and `validate_chain` return issues through `classify`: an expressibility gap of the expression language or its vocabulary has severity `gap` (D560); `errors_of` ignores it, so the object is admitted (status `proposed`). Today `bad_expression` for an unsupported construct (not a typo) and `undeclared_pattern_param` are gaps.

**Admit** (`kta.knowledge.admission`, D561).

```python
warning_key(issue, file) -> str                        # "<kind>@<file>:<where>"
load_explanations(path) -> dict[str, str]              # warnings.yaml
admit(pairs: Iterable[(file, Issue)], explanations) -> Admission   # errors, warnings, language_gaps
admit_references(ctx, store, root, code=None) -> (ImportResult, {reference name: Report ID})
latest_admissions(store, reference_ids) -> {reference ID: admission data}
```

A reference directory (the decompose-kernel layout) may hold `warnings.yaml` mapping warning keys to explanations. `Admission.admitted` is false when there is an error or an unexplained warning. `admit_references` runs `import_references(..., admit=...)`: errors or unexplained warnings refuse the whole batch before anything is written; language gaps are counted. Each admitted reference gets a Report `title: admission` (`subject`: its Reconstruction, `refs.reference`, `data`: `Admission.to_dict()`). `ImportResult.language_gaps` lists the gap issues of a batch.

**Coverage additions** (`kta.knowledge.sota`, D567). `CoverageReport.learning` and the page's `learning` field hold `sota_levels` (one row per level: references at that level or above, `of`, `pending` reason; level 1 also `step_rate`), `system_features` (`total`, `by_feature`), `language_gaps` (`total`, `by_kind`, `by_reference`, from the latest admission Report per reference), and `backends` (per backend: `steps`, `expressible` = steps whose Template has code on that backend from a Realisation with code markers or fragments or a Family chain item with code markers, `rate`, `intent_declared` = steps with SASS expectations on it or `any`, `intent_confirmed` = null until compiled SASS exists). Level 2 counts Reconstructions with `reproduction_error ≤ ε`; levels 3–4 count Reports `title: o12` (`data.reference`, `data.level`).

**Ablation** (`kta.knowledge.ablation`, D566).

```python
factors(family, chain) -> list[Factor]                 # chain steps and variables with ≥ 2 values
suspected_pairs(factors, family, chain, templates) -> list[Pair]   # shared_variable | template_relation | shared_resource
design(factors, pairs, controls=None) -> Design                       # pairwise (base, singles, pairs) or fractional 2^(k-p)
estimate(design, times, noise_band=None, controls=None) -> list[Estimate]          # joint / product of single effects
propose_couplings(store, design, estimates, *, family_id, target, workload, evidence, actor, now) -> list[str]
```

A coupled pair (|joint/product − 1| > noise band) becomes a Coupling record (`basis: measured`, `refs.family`, `refs.evidence` = the run measurements), status `proposed`.

**Reference schema 2.** `source_class` (open name, default `open_code`) and `priority` (`ReferencePriority`); schema-1 records upcast with those defaults (`kta.core.schemas.upcast.reference_v1`).
