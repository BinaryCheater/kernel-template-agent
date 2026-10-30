# knowledge

设计依据：docs/design/21、22、23、24。schema 见 spec/schemas.md §4。

## 文件

```
knowledge/patterns.py      Pattern 表的加载、新增、匹配
knowledge/templates.py     Template 的加载、校验、查重、版本
knowledge/chains.py        Chain 的校验；链项之间的关系检查
knowledge/families.py      族目录的加载与一致性检查
knowledge/scope.py         Γ 求值；增量复用的逐步检查
knowledge/retrieve.py      检索：Template、族、链、按诊断、差集
knowledge/render.py        卡片、族概览、文字知识渲染
knowledge/coverage.py      覆盖统计（21 §2）
knowledge/seed/            初始 Pattern 表、Structure 表、论文 GEMM 五步的 Template（YAML）
```

## 接口

```python
# Pattern
def load_patterns(store) -> dict[str, Pattern]
def match_pattern(tx: TxDescription, patterns) -> list[tuple[str, float]]   # 候选模式及匹配度；无匹配返回 [("other", 0)]

# Template
def validate_template(t: Template, store, hw: dict) -> list[Issue]
    # 字段完整；表达式可解析、自由变量已声明；preconditions 的 same_as 指向存在的约束；
    # relations 的 target 存在；refines 和 produces 的 Pattern 存在；requires_hw 的 HwFeature 存在；
    # 查重：同一 refines、同类 object.type、变量名集合与成本单元集合相同 → Issue(kind="possible_duplicate")
def propose_template(ctx, t: Template, sources: list[str]) -> str     # 校验通过后写入，状态 proposed，保证等级 declared

# Chain 与 Family
def validate_chain(ch: Chain, store) -> list[Issue]
    # 每个 item 的 template 存在；bindings 覆盖该 Template 的全部变量；after 构成 DAG；
    # 链中 Template 的 requires/excludes 关系满足
def load_family(path: Path) -> FamilyPackage
def validate_family(fp: FamilyPackage, store, hw: dict) -> list[Issue]
    # validate_chain；每个 item 的 code_markers 都在 gen.py 生成的源码中出现（对一组默认取值生成后检查）；
    # 族变量 = 各 item 绑定到的变量 ∪ glue 中的变量，没有多余也没有缺失；
    # glue_constraints 每条有 meaning；generate() 对边界形状能正常返回

# 适用范围与增量复用
def scope_holds(t: Template, case: Case, hw: dict, params: dict) -> ScopeResult    # holds: bool | None（无法求值）、reason
def reuse_plan(chain_id: str, case: Case, hw: dict) -> ReusePlan
    # 对链中每一步求值 Γ；返回 keep / invalid / unknown 三类及原因；
    # 对 invalid 的步骤，给出 alternative_of 的候选

# 检索
def retrieve_families(case: Case, k: int = 5) -> list[Card]
def retrieve_templates(pattern: str | None, unit: str | None, text: str | None, case: Case | None, k: int = 10) -> list[Card]
def nearest_chains(case: Case, k: int = 3) -> list[Card]
def diff_templates(reference_chain: str, current_chain: str) -> list[Card]     # 差集检索
def card(record_id: str) -> Card          # 五行以内：名称、作用、强弱项（来自证据）、状态、保证等级

# 渲染与覆盖
def family_overview(pattern_or_problem: str) -> str     # Markdown，一页
def render_text_knowledge(snapshot_id: str, out_dir: Path) -> None
def coverage_report(snapshot_id: str) -> CoverageReport   # 21 §2 的全部量
```

## 行为

- **开放的表。** Pattern、HwFeature、Structure 都存在存储中，可以随时新增。代码中不得写死 Pattern 的枚举；`category` 的取值是 schema 中固定的少数几种，含 `other`。
- **查重是提示，不是拒绝。** `possible_duplicate` 会写入 Issue 和 Submission 的检查结果；worker 必须在交付说明中解释区别，否则完成检查不通过。
- **检索排序**：先硬过滤（Pattern 匹配、Γ 成立或无法求值、requires_hw 在当前 Target 上为 yes），再按以下顺序排序：保证等级、在相似 case 上的 %下界（来自 Solution）、复用次数。Γ 无法求值的排在后面，并在卡片上标明。
- **卡片和概览只从证据生成**，不读取作者写的评价性文字（summary 除外，且标明是作者说明）。
- **覆盖统计**（`CoverageReport`）至少包括：
  - Template 的发现曲线（按 Reconstruction 的时间顺序，随机重排 50 次）、Chao1 估计、Good–Turing 覆盖率；
  - 各 Reference 的匹配率、未匹配事务数（按 level 和 pattern_guess 统计）；
  - Pattern 中 `other` 的数量；
  - HwFeature × Pattern 网格；
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
