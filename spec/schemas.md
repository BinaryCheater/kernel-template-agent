# 记录与知识的 schema

所有 schema 用 pydantic v2 实现，放在 `src/kta/core/schemas/`。下文用类 Python 写法表示字段；`?` 表示可选，`list[...]` 默认为空列表。字段名即代码中的名字。

## 1. 记录信封

每条证据记录都包在信封里（docs/design/20 §2）：

```python
class Actor:
    type: Literal["tool", "checker", "agent", "human", "system", "import"]
    name: str                 # 例如 "kta measure"、"numeric_checker@3"、"worker"
    task: str?                # 产生它的任务 ID（agent 或工具在任务中调用时）
    commit: str?              # git commit of the code that wrote the record (D84, D140)
    dirty: bool?              # uncommitted changes in that code; set together with commit

class Record:
    kind: str                 # 见 §3、§4 的 kind 名
    schema: int               # 该 kind 的 schema 版本，从 1 开始
    created_at: datetime      # UTC，ISO 8601
    actor: Actor
    refs: dict[str, str | list[str]]   # 指向其他记录的 ID；键名由各 kind 规定
    body: dict                # 该 kind 的字段（§3、§4），经 pydantic 校验
    blobs: dict[str, str]     # 名字 → 产物 sha256
    origin: Literal["real", "fixture", "simulated", "historical"] = "real"
```

- (D141) In Python the field `schema` is `Record.schema_` (JSON alias `schema`); the stored JSON uses `schema`. (D144) The body is stored normalized (every default filled in) and `created_at` in UTC before hashing. (D140) `Store.put` fills `actor.commit`/`actor.dirty` from its code state when absent and keeps them when present.
- **ID** = `sha256(canonical_json(record))` 的十六进制。`canonical_json`：键排序、无多余空白、UTF-8、datetime 用 ISO 8601、浮点用 `repr`。
- `origin` 必填。只有 `real` 的记录能进入正式结论。
- `refs` 中的每个 ID 必须已存在于存储中，否则写入失败。
- 由作业产生的记录（Build、Check、Measurement、Profile）和 AgentRun，`refs` 中必须有 `env`（EnvManifest）；作业产生的记录另有 `worker`（Worker）。

## 2. 通用取值

```python
EvidenceStatus = Literal["proposed", "reproduced", "calibrated", "holdout_validated", "rejected", "deprecated", "stale", "suspect"]
Assurance      = Literal["declared", "tested", "bounded", "proved"]
Level          = Literal[0, 1, 2, 3, 4, 5]     # 论文的细化层级：0 计算图，1 算子变换，2 张量执行映射，3 指令调度，4 系统架构，5 微架构
ChangeKind     = Literal["vertical", "expansion", "horizontal", "restriction"]
Binding        = Literal["compile", "runtime"]
VarRole        = Literal["software", "hardware", "boundary"]
VarKind        = Literal["structure", "numeric"]
Unit           = str      # 硬件单元名，取自 Target 的单元表，例如 "dram", "l2", "smem", "tensor", "fma", "issue", "launch"
```

状态与保证等级由 StatusChange 记录推导（§4.9），不写在对象本身的 body 里。

## 3. 表达式

约束、前提、Γ、成本项都用同一种表达式语言：Python 表达式的一个子集，以字符串存储，由 `kta.core.expr` 解析和求值。

- 允许：数字和布尔字面量；变量名；`+ - * / // % **`；比较；`and or not`；`if ... else ...`；函数 `min max abs ceil floor ceil_div log2 is_pow2`；属性访问 `hw.<名>`（Target 的硬件契约）、`p.<名>`（模式或 workload 参数）、`case.<名>`。
- 不允许：其他函数调用、下标、lambda、推导式、import。
- String literals (added by D97): only as operands of `==` / `!=`, and as elements of a non-empty literal list on the right of `in` / `not in` (e.g. `mechanism == "cp.async"`, `level in ["smem", "reg"]`). Ordering comparisons and arithmetic on strings are not allowed; any comparison with a string literal is non-linear.
- 解析时检查：所有自由变量都在声明的变量、`hw`、`p`、`case` 之中。
- **线性子集**：只含加减、常数乘、比较、`and` 的表达式被标记为线性，可以导出到 MIP；其余由枚举求解器处理。

## 4. 知识对象

### 4.1 Pattern（事务模式）

```python
class Pattern:
    name: str                 # 唯一，例如 "reduce", "transfer"
    level: Level
    category: Literal["reduce", "contract", "map", "store", "transfer", "sync", "access", "other"]
    params: list[ParamSpec]   # 模式参数，例如 reduce 的 assoc、comm、rescalable
    description: str          # 两三句
    examples: list[str]       # 至少两个
class ParamSpec:
    name: str; type: Literal["bool", "int", "float", "str", "enum"]; values: list[str]?; meaning: str
```
refs：无。`category = "other"` 的 Pattern 允许存在，用于暂存描述不了的事务。

### 4.2 Template（细化步骤）

```python
class Template:
    name: str                       # 例如 "async_transfer"
    version: int
    level_from: Level; level_to: Level
    change: ChangeKind
    refines: str                    # Pattern name
    produces: list[str]             # Pattern names
    object: ObjectSpec              # O
    variables: list[VarSpec]        # X
    constraints: list[ExprSpec]     # C
    preconditions: list[ExprSpec]   # Pre；每条可用 same_as 指向某条 C 的 id（表示 K1 逐字成立）
    program: ProgramSpec            # P
    backward_map: str               # π：文字说明；可选 python 函数路径 backward_map_fn
    obligations: list[Obligation]   # Σ
    cost: list[CostTerm]            # Ĵ
    scope: ScopeSpec                # Γ
    requires_hw: list[str]          # HwFeature names
    relations: list[Relation]
    signature: SignatureSpec        # ς
    numeric_risk: str?              # 若改变数值：说明；检查时必须加入的输入分布名
    expected_sass: list[SassExpect]
    summary: str                    # 三句以内的说明，给人和 agent 读
    example_code: str?              # blob 名，最小代码示例

class ObjectSpec:   type: str; description: str          # 例如 type="smem_slot"
class VarSpec:      name: str; kind: VarKind; binding: Binding; role: VarRole
                    domain: list[int|str|bool] | RangeSpec; unit: str?; meaning: str
                    when: str?                            # 生效条件表达式
class RangeSpec:    lo: int; hi: int; step: int = 1; pow2: bool = False
class ExprSpec:     id: str; expr: str; meaning: str
                    cls: Literal["semantic", "resource", "implementation", "performance_assumption"]
                    same_as: str?                         # 仅 preconditions 使用
class ProgramSpec:  description: str; fragments: dict[str, str]   # 后端名 → blob 名（参考代码片段）
class Obligation:   name: str; statement: str
                    method: Literal["test", "exhaustive", "smt", "proof", "unspecified"]
class CostTerm:     unit: Unit; work: str                 # 表达式：该单元上的工作量（字节、FLOP、次数）
                    overlap: Literal["overlappable", "serial"] = "overlappable"
class ScopeSpec:    predicate: str                        # 表达式，可引用 p.*、hw.*、case.*
                    text: str
class Relation:     type: Literal["requires", "excludes", "enables", "refines", "alternative_of"]
                    target: str                           # Template name
                    note: str?
class SignatureSpec: algorithm: str; reuse: str?; storage: str?; boundary: str?; resources: list[str]
class SassExpect:   when: str; must_contain: list[str]; must_not_contain: list[str]
```
refs：`previous`（上一版本，可无）、`sources`（Source 列表，至少一个，除非 actor 为 human）。

### 4.3 Chain（组合链）

```python
class Chain:
    workload: str                   # 一句话：这条链从什么 workload 出发
    items: list[ChainItem]
class ChainItem:
    key: str                        # 链内唯一名，例如 "k_split"、"a_async"
    template: str                   # "<name>@<version>"
    applies_to: str                 # 细化的是哪个具体事务
    bindings: dict[str, str | int | bool]   # Template 变量名 → 族变量名或常量
    after: list[str]                # 依赖的前序链项 key；构成 DAG
    code_markers: list[str]         # 生成器中的标记名；重建链可为空
```
refs：`templates`（全部用到的 Template 记录 ID）。

### 4.4 Family（可执行族）

```python
class Family:
    name: str; version: int
    backend: Literal["tilelang", "cuda"]
    variables: list[VarSpec]        # 族变量
    glue_constraints: list[ExprSpec]   # 每条的 meaning 必须说明为什么不能归入某一步
    dispatch: DispatchSpec?
    edge_cases: list[dict]          # 边界形状
    numeric_risk: list[str]         # 汇总
    entry: str                      # 生成器入口，例如 "gen.py:generate"
class DispatchSpec: runtime_vars: list[str]; buckets: list[dict]; rule: str
```
refs：`chain`、`previous`、`problem`。blobs：`package`（族目录的 tar）。

族目录结构（`kta validate-family` 检查）：

```
family.yaml        # Family 的 body
chain.yaml         # Chain 的 body
gen.py             # generate(assignment: dict, case: dict) -> GeneratedKernel
edge_cases.yaml
README.md          # 一页以内
```

`GeneratedKernel` 见 spec/modules/target-exec.md。

### 4.5 Reference 与 Reconstruction

```python
class Reference:
    name: str; problem: str; build: str     # 构建方式（命令或脚本）
    runnable_on: list[str]                  # Target 名；空表示仅分析
class Reconstruction:
    matched: int; total: int                # 匹配的链项数 / 全部步骤数
    unmatched_transactions: list[UnmatchedTx]
    reproduction_error: float?              # 复现性能相对误差
    notes: str
class UnmatchedTx: level: Level; pattern_guess: str; description: str; where: str   # 源码或 SASS 位置
```
Reference refs：`source`、`problem`。Reconstruction refs：`reference`、`chain`、`gaps`、`measurement`（可无）。

### 4.6 硬件

```python
class Target:     name: str; gpu: str; sm: str; driver: str; cuda: str; ptxas: str
                  image: str; backends: dict[str, str]     # 后端名 → 版本或 commit
class HwFeature:  name: str; unit: Unit; instructions: list[str]; description: str
                  support: dict[str, Literal["yes", "no", "unknown"]]   # Target 名 → 是否支持
class HwFact:     feature: str?; unit: Unit?; quantity: str; value: float | str
                  ci_low: float?; ci_high: float?; method: Literal["measured", "documented"]
class Worker:     name: str; host: str; transport: Literal["local", "ssh", "pull"]
                  labels: dict[str, str]                 # 能力标签，例如 gpu=sm_121、cuda=13.0、pool=gpu-exclusive
                  devices: list[str]; commit: str
class EnvManifest: role: Literal["control", "worker"]; host: str; os: str; arch: str
                  python: str; uv: str; uv_lock_sha256: str; node: str?; npm: str?; npm_lock_sha256: str?
                  cuda: str?; driver: str?; nvcc: str?; ptxas: str?; backends: dict[str, str]
                  image_digest: str?; commit: str; dirty: bool; lock_mismatches: list[str]
class HwModel:    target: str; version: int; peaks: dict[Unit, float]; limits: dict[str, int | float]
                  # limits 即表达式中的 hw.*，例如 smem_per_cta、regs_per_thread、max_threads_per_cta
```

### 4.7 其他知识

```python
class Structure:  name: str; effect: Literal["objective", "variables", "enumeration"]; description: str; evidence_needed: str
class Regime:     structure: str; context: str; holds: bool; how: Literal["proved", "observed"]
class Negative:   subject: str; condition: str; effect: str
class Gap:        missing: Literal["pattern", "template", "relation", "hw_fact", "family", "model_term", "check", "other"]
                  description: str; level: Level?; pattern_guess: str?
                  attribution: Literal["space", "model", "intent_sass", "verification", "process", "granularity", "other"]
```

### 4.8 求解与评价

```python
class Problem:   name: str; reference_impl: str; inputs: list[TensorSpec]; outputs: list[TensorSpec]; params: dict
class TensorSpec: name: str; shape: list[str]; dtype: str; layout: str
class Contract:  tolerances: dict[str, dict]          # dtype → {"rtol":…, "atol":…, "rel_l2":…}
                 input_dists: list[str]; edge_shapes: list[dict]; timing_boundary: str
class Case:      problem: str; shape_dist: list[dict]  # 每项 {"shape": {...}, "weight": w}
                 objective: Literal["weighted_latency", "max_normalized_latency"]; target: str
class Candidate: assignment: dict; bucket: str?
class Prediction: per_unit: dict[Unit, float]; serial: float; total: float; model: str
class Build:     ok: bool; regs: int?; spills: int?; smem: int?; log_summary: str
class Check:     passed: bool; failures: list[dict]    # 每项：检查名、输入分布、形状、最大误差等；不含隐藏形状本身
class Measurement: median_us: float; ci_low_us: float; ci_high_us: float; n: int
                 clocks: dict; temps: dict; protocol: str
class Profile:   metrics: dict[str, float]
class Bound:     per_unit_us: dict[Unit, float]; bound_us: float; bottleneck: Unit
class Solution:  bucket: str?; assignment: dict; pct_of_bound: float
```

### 4.9 流程与审计

```python
class StatusChange: subject: str; to_status: EvidenceStatus | None; to_assurance: Assurance | None; reason: str
class TaskCard:     type: Literal["T1", "T2", "T3", "T4", "T5", "T6"]; goal: str; inputs: list[str]
                    done_checks: list[str]; budget: dict; profile: str; work_unit: str   # profile：AgentProfile 记录 ID
class AgentRun:     session: str; profile: str; cli_version: str; model: str
                    tokens_in: int?; tokens_out: int?           # 会话中带有用量时由解析器填写；否则为空
                    session_archive: str                      # blob：该会话的会话文件与 CLI 输出（不含凭据）
                    wall_s: float; tool_s: float; queue_s: float; exit: str
                    context_manifest: str             # blob：本次运行全部上下文来源及其 hash（agents.md）
                    violations: list[dict]            # 越权尝试、未允许的命令、工作区外写入等
                    arm: dict                         # experiment arm: name, knowledge (zero | text | pool), snapshot (D430)
class AgentProfile:                                   # 控制 agent 行为的全部设置（agents.md §受控配置）
    name: str; version: int
    provider: Literal["codex", "claude_code", "pi", "deepseek_harness"]; cli_version: str
    auth: Literal["subscription", "api"]
    api_base: str?; api_key_name: str?   # API 方式时：provider 的 API 地址；key 在机密文件中的名字（不存 key 本身）
    isolation: Literal["config", "config+user", "container"]   # 隔离程度，见 docs/decisions.md D09
    model: str; reasoning_effort: str?
    cli_flags: list[str]                  # 沙箱、审批、关闭 web search 与 MCP 等
    home_files: dict[str, str]            # 隔离的 CLI 配置目录（CODEX_HOME）中的文件名 → blob
    prefix_template: str                  # blob：worker 工作区 AGENTS.md 的模板
    step_templates: dict[str, str]        # 任务类型 → blob：TASK.md 模板
    message_templates: dict[str, str]     # rejection、budget_warning、human_note、new_session → blob
    skills: dict[str, list[str]]          # 任务类型 → ["<skill 名>@<hash>"]
    skill_bundles: dict[str, str]         # "<skill 名>@<hash>" → blob (deterministic tar of the skill dir; D322)
    allowed_commands: dict[str, list[str]]   # 任务类型 → 允许的 kta 子命令
    context_policy: dict                  # cards_k、card_max_lines、tool_output_max_chars、step_header_max_chars、notes_required
    loop_policy: dict                     # max_retries、reset_context_ratio、reset_on_repeat_failure、idle_s、各任务类型的预算
                                          # WP21 (D483): both kept as recorded; runs read these knobs from the controls registry
class Submission:   files: list[str]; note: str          # 交付说明
class HumanAction:  action: str; reason: str
class PoolSnapshot: members: list[str]; note: str
```

### 4.10 Kinds listed in docs/design/20 §2.3 without fields above (D143, needs user review)

```python
class Source:        name: str; type: Literal["git", "archive", "paper", "doc", "other"]
                     locator: str            # URL or path it was obtained from
                     revision: str           # commit, archive sha256, DOI or edition
                     license: str?; note: str?
class Protocol:      name: str; version: int; type: Literal["measurement", "experiment", "other"]
                     description: str; params: dict
class ExperimentRun: name: str; condition: str; seed: int?; params: dict; note: str
class Report:        title: str; subject: str; summary: str; data: dict
class ControlsVersion:  version: int (>= 1); definitions_hash: str   # WP21, D481
                     values: dict              # knob id -> value, every knob of the definitions
                     changes: list[dict]       # {knob, old, new} against refs.previous
                     reason: str               # required; the actor is the author
```
refs: Report may have `env` (EnvManifest). ControlsVersion has `previous` (ControlsVersion) except version 1. TaskCard and AgentRun may have `controls` (the ControlsVersion the card's budget / the step used), ExperimentRun `controls` (a list: the versions its loop ran under) (WP21). Template's `backward_map_fn: str?` (python function path) is a field of its own.

推进规则（`kta.store.status`）：

| 变更 | 允许的 actor |
| --- | --- |
| 任何对象 → proposed（创建时默认） | 任何 |
| proposed → reproduced / calibrated / holdout_validated | checker |
| 任何 → rejected / deprecated / suspect | human |
| 任何 → stale | system（依赖的记录变更或被驳回时，见 docs/design/20 §6） |
| 保证等级 declared → tested / bounded / proved | checker |

不满足规则的 StatusChange 写入时被拒绝。

Interpretation used by the implementation (D149, needs user review): checkers may also promote from `stale` and forward between promotion levels; `-> proposed` is allowed for any actor only while the object is proposed, for a human from any status; assurance only moves up.
