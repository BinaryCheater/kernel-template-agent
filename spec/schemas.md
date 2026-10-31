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
- (D440) **Schema versions and upcast.** A kind's `schema` is `1 + len(upcasters)` (`kta.core.schemas.registry.KindSpec`); `upcasters[n-1]` maps a stored body of version n to n+1. A record stored under an older version is read with its stored body kept for hashing (`Record.to_json_dict()`, `Record.stored_body`) and its `body` upcast to the current shape; `schema_` stays the stored version and `Record.upcast` is true. A version newer than the code is refused. Stored records are never rewritten. Reference 2 (WP20): `source_class` and `priority` added with defaults (D564). Current versions changed by WP14: Pattern 2, Template 2, Gap 2, Reconstruction 2 (levels 0–5 → 1–7; Template also `requires_hw` → `requires`, `probed_on` added, `program.fragments` and `expected_sass` moved to Realisation).
- **ID** = `sha256(canonical_json(record))` 的十六进制。`canonical_json`：键排序、无多余空白、UTF-8、datetime 用 ISO 8601、浮点用 `repr`。
- `origin` 必填。只有 `real` 的记录能进入正式结论。
- `refs` 中的每个 ID 必须已存在于存储中，否则写入失败。
- 由作业产生的记录（Build、Check、Measurement、Profile）和 AgentRun，`refs` 中必须有 `env`（EnvManifest）；作业产生的记录另有 `worker`（Worker）。
- (D461) WP15 appends to D440's mechanism: Template 3 (`template_v2_effects`: `cost` → `effects`; the Template model also upcasts a `cost` body itself), HwModel 2 (class + instance fields, all optional), Prediction 2 (`bounds`, `unpriced`); both of the latter are identity steps.

## 2. 通用取值

```python
EvidenceStatus = Literal["proposed", "reproduced", "calibrated", "holdout_validated", "rejected", "deprecated", "stale", "suspect"]
Assurance      = Literal["declared", "tested", "bounded", "proved"]
Level          = Literal[1, 2, 3, 4, 5, 6, 7]  # (D441) 1 workload, 2 graph, 3 operators, 4 tensor map, 5 instructions, 6 architecture, 7 microarchitecture
ChangeKind     = Literal["vertical", "expansion", "horizontal", "restriction"]
Binding        = Literal["compile", "runtime"]
VarRole        = Literal["software", "hardware", "boundary"]
VarKind        = Literal["structure", "numeric"]
Unit           = str      # 硬件单元名，取自 Target 的单元表，例如 "dram", "l2", "smem", "tensor", "fma", "issue", "launch"
```

(D441) Records written before WP14 used the paper's 0–5 (0 graph, 1 operator transforms, 2 tensor execution mapping, 3 instruction scheduling, 4 system architecture, 5 microarchitecture). They are read through the named mapping `paper-0to5@1` in `kta.core.schemas.levels` (n → n + 2). A published mapping is never edited; a later renumbering adds a new name.

状态与保证等级由 StatusChange 记录推导（§4.9），不写在对象本身的 body 里。

## 3. 表达式

约束、前提、Γ、成本项都用同一种表达式语言：Python 表达式的一个子集，以字符串存储，由 `kta.core.expr` 解析和求值。

- 允许：数字和布尔字面量；变量名；`+ - * / // % **`；比较；`and or not`；`if ... else ...`；函数 `min max abs ceil floor ceil_div log2 is_pow2`；属性访问 `hw.<名>`（Target 的硬件契约）、`p.<名>`（模式或 workload 参数）、`case.<名>`。
- 不允许：其他函数调用、下标、lambda、推导式、import。
- String literals (added by D97): only as operands of `==` / `!=`, and as elements of a non-empty literal list on the right of `in` / `not in` (e.g. `mechanism == "cp.async"`, `level in ["smem", "reg"]`). Ordering comparisons and arithmetic on strings are not allowed; any comparison with a string literal is non-linear.
- 解析时检查：所有自由变量都在声明的变量、`hw`、`p`、`case` 之中。
- (D462) Hardware-model expressions (spec §4.6.1) use the same language with other attribute roots: `hw.<name>` (instance values), `tx.<name>` (per-transaction amounts, in reservation tables), `f.<name>` (features such as `f.residency`, in efficiency and `applies_when`), `tbl.<name>` (table terms, looked up before evaluation). `Expr.parse(src, declared, roots)` and `Expr.eval(..., extra={root: scope})`; Template expressions keep the roots `hw`, `p`, `case`.
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
    effects: Effects                # Ĵ, in the transaction vocabulary only (D460; was cost: list[CostTerm], schema 1)
    scope: ScopeSpec                # Γ
    requires: list[str]             # Capability names (abstract; D442). Was requires_hw (HwFeature names) in schema 1
    probed_on: list[str]            # Target names where a family using it compiled and ran (D445); [] = unprobed
    relations: list[Relation]
    signature: SignatureSpec        # ς
    numeric_risk: str?              # 若改变数值：说明；检查时必须加入的输入分布名
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
class ProgramSpec:  description: str                  # the abstract program; code lives in Realisations (D443)
class Obligation:   name: str; statement: str
                    method: Literal["test", "exhaustive", "smt", "proof", "unspecified"]
class CostTerm:     unit: Unit; work: str                 # schema 1 only; Template.cost is now a read-only legacy view
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

#### 4.2b Realisation (D443)

How a Template is written on one backend over an ISA range.

```python
class Realisation:
    template: str                   # "<name>@<version>"
    backend: str                    # open name: tilelang, cuda, triton, ...; "any" = backend-independent
    isa: IsaRange
    requires: list[str]             # ISA features (HwFeature names) it uses, e.g. cp_async
    isa_constraints: list[ExprSpec] # constraints the ISA adds over the Template's variables
    fragments: dict[str, str]       # fragment name -> blob (reference code)
    code_markers: list[str]
    expected_sass: list[SassExpect] # when names Template variables
    probed_on: list[str]            # Target names
    summary: str
class IsaRange:     family: str; min: str?; max: str?          # inclusive, e.g. family "sm", min "80"
```
refs: `template` (required, Template), `sources`, `previous`. A schema-1 Template's `program.fragments` and `expected_sass` are read back as one legacy realisation (backend `any`, no record) by `kta.knowledge.realisations`.
#### 4.2.1 Effects (D460, D461; docs/design/02 §3)

```python
class Effects:
    transactions: list[TxEffect]
    storage: list[StorageEffect]
    overlap: list[OverlapEffect]
    upcast_from: str?               # "cost" when produced from a schema-1 cost list
class TxEffect:      id: str; pattern: str          # transaction pattern (open table)
                     kind: str                      # operation class of the hardware model (open; unknown -> unpriced)
                     kind_map: dict[str, str]       # when kind names a variable / p.*: binding value -> class
                     count: str                     # expression; transactions per instance of the refined transaction
                     amounts: dict[str, str]        # per transaction: bytes, flops, wavefronts, ops, warps, ...
                     src: str?; dst: str?           # storage levels of a transfer
                     when: str?; note: str?         # the transactions exist only when `when` holds
class StorageEffect: id: str; level: str            # level name, or a variable / p.* whose binding names it
                     scope: str = "cta"; bytes: str; when: str?; note: str?
class OverlapEffect: between: list[str]             # 1-2 selectors: a local tx id, "pattern:<p>", "kind:<k>"
                     when: str?                     # they overlap while it holds (always when absent), serialise otherwise
                     inflight: str?; trips: str?    # with both: recurrence bound trips * (lat_a + lat_b) / inflight
                     note: str?
```

- No hardware number appears in effects. Ids are unique within a Template; a one-selector overlap means "with everything else" (`when: "False"` makes a transaction serial).
- **Declared once** (D460): each physical transaction is declared by the step that performs it (operand traffic by the transfer step, the contraction by the instruction-selection step, compulsory DRAM traffic and launches by the tiling step). A coarser step does not repeat what a finer step declares.
- Chain bindings rewrite effects like constraints; `p.instances` multiplies every `count` (not storage, which is per executor instance).
- Upcast of schema 1: cost term `i` becomes `{id: cost<i>, pattern: other, kind: "unit:<unit>", count: <work>}`, and a `serial` term adds `{between: [cost<i>], when: "False"}`. The built-in `unit:<u>` class reserves one unit of resource `u` per count, so an upcast Template prices exactly as before.

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
    source_class: str = "open_code"         # schema 2 (WP20): open name, learning.yaml source_classes
    priority: ReferencePriority?            # schema 2: cases_affected, gap_closable, uncertainty, reasons
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
class Capability: name: str; description: str; unit: Unit?          # abstract capability (D442)
                  params: list[ParamSpec]                            # what a declaration states, e.g. max_cluster_size
                  realised_by: list[str]                             # ISA features (HwFeature names) that provide it
class CapabilityDecl: target: str; capability: str                   # one Target's declaration; newest per pair wins
                  status: Literal["yes", "no", "unknown"]; params: dict[str, int | float | str | bool]
                  basis: Literal["documented", "probed", "inferred"]; note: str
class HwFeature:  name: str; unit: Unit; instructions: list[str]; description: str   # ISA instruction family (probe)
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
                  # schema 2 (D463): the chip instance of the hardware model
                  arch: str?                              # "<HwArch name>@<version>"; absent = schema-1 flat model
                  provenance: dict[str, Provenance]       # "peaks.<r>" / "limits.<name>" -> source; every value when arch is set
                  tables: list[TableSpec]; efficiency: list[EfficiencySpec]; interactions: list[InteractionSpec]
                  # peaks = measured chip-level amount/s per resource; they override the class's peak expressions
```
refs (HwModel): `arch` (HwArch), `previous_instance` (HwModel), plus WP7's `target`, `facts`, `measurements`, `protocol`.

#### 4.6.1 Declarative hardware model (D462–D465, D469–D471; docs/design/02 §4)

```python
ValueStatus = Literal["measured", "documented", "default"]
class Provenance:     status: ValueStatus; source: str; fact: str?        # fact = HwFact record ID
class HwArch:         name: str; version: int; description: str           # the architecture class
                      scopes: dict[str, str]                               # scope -> instance count over hw.* (chip, sm, smsp, ...)
                      params: list[str]                                    # hw.* names every instance must give
                      resources: list[ResourceSpec]; walls: list[WallSpec]
                      operation_classes: list[OperationClass]; tables: list[TableSpec]
                      efficiency: list[EfficiencySpec]; interactions: list[InteractionSpec]
                      composition: CompositionSpec
class ResourceSpec:   name: str; kind: str; scope: str; amount: str        # kind: throughput | capacity | latency (open)
                      peak: str?                                           # amount/s per scope instance over hw.*; absent = unmodelled
                      latency_cycles: str?; max_inflight: str?; ncu: list[str]; source: str
class WallSpec:       name: str; kind: str; level: str; scope: str; limit: str; reserved: str?
                      cliff: str                                           # no_launch | spill | lower_occupancy | other (open)
                      per: str?; source: str
class OperationClass: name: str; patterns: list[str]                       # name = TxEffect.kind
                      reservation: list[ReservationRow]                    # at least one row
                      latency_cycles: str?; serial: bool = False; sass: list[str]; source: str
class ReservationRow: resource: str; amount: str; latency_cycles: str?     # amount over tx.*, hw.*, tbl.*
class TableSpec:      name: str; keys: list[str]; rows: list[list[float]]  # keys: "tx.x" / "f.x" / "hw.x"; row = keys..., value
                      lookup: Literal["exact", "floor"] = "exact"; source: str
class EfficiencySpec: resource: str; expr: str; features: list[str]        # expr over f.*, tbl.*, hw.*, value in (0, 1]
                      status: ValueStatus; source: str; fit_data: list[str]; holdout_error: float?
class InteractionSpec: name: str; resources: list[str]; gamma: float        # adds gamma * min(busy_a, busy_b)
                      applies_when: str = "True"; evidence: list[str]; status: ValueStatus; source: str
class CompositionSpec: rule: Literal["steady_state_max"]; clock_hz: str = "hw.sm_clock_hz"; note: str
class ModelVersion:   target: str; label: str; hw_model: str; arch: str?   # immutable model version (02 §4 Maintenance)
                      simulators: dict[str, str]                           # fidelity level -> code commit
                      change: Literal["initial", "adapter_fix", "machine_model_change"]; reason: str
                      results: list[RegressionResult]; summary: dict[str, float]; admitted: bool?
class RegressionResult: kernel: str; predicted_us: list[float]; measured_us: list[float]
                      rank_correlation: float?; error_by_resource: dict[str, float]   # "total" | <resource> (D507)
                      holdout: bool = False; predictions: list[str]; measurements: list[str]   # per configuration, "" = none (D507)
class RegressionSet:  name: str; version: int; target: str; min_configurations: int = 2   # >= 2: ranking needs several
                      cases: list[RegressionCase]
class RegressionCase: kernel: str; shape: dict; configurations: list[dict]; measurements: list[str]; holdout: bool = False
# artefact (blob) schema, not a record kind:
class InterferenceMatrix: target: str; arch: str?; protocol: str?; classes: list[str]; cells: list[InterferenceCell]
class InterferenceCell: a: str; b: str; status: Literal["measured", "declared_independent", "unmeasured"]
                      alone_a_s: float?; alone_b_s: float?; together_s: float?; slowdown: float?; measurements: list[str]; note: str?
```

- HwArch validation refuses duplicate names, a reservation, efficiency or interaction naming an undeclared resource, a resource scope not in `scopes`, and any expression outside its roots (scopes, peaks, limits, latencies: `hw`; reservation amounts: `hw`, `tx`, `tbl`; efficiency and `applies_when`: `hw`, `f`, `tbl`).
- Unmodelled = infinite capacity (D465): a resource without a peak, or an operation class without a reservation table, costs no time and is reported (`L1Result.unmodelled`, `L1Result.unpriced`, `Prediction.unpriced`).
- refs: HwArch `previous`, `sources`; ModelVersion `hw_model`, `arch`, `regression_set`, `previous`; RegressionSet `previous`.

### 4.7 其他知识

```python
class Structure:  name: str; effect: Literal["objective", "variables", "enumeration"]; description: str; evidence_needed: str
class Finding:    claim: str; kind: str                       # (D444) open kind: regime, negative, positive, observation, other
                  subject: str?                               # Template / Family / Structure / Capability name
                  scope: Scope; holds: bool = True
                  basis: Literal["observed", "measured", "proved", "claimed"]
                  explained_by: list[str]                     # record IDs or "HwModel:<id>#<term>"
class Scope:      targets: list[str]; workload: str = ""; condition: str = ""; predicate: str?   # empty = unrestricted
class Coupling:   members: list[CouplingMember]               # (D447) at least two
                  scope: Scope; effect: str; joint_ratio: float?; noise_band: float?
                  basis: Literal["observed", "measured", "proved", "claimed"]
class CouplingMember: kind: Literal["variable", "template", "resource", "other"]; name: str
class Regime:     structure: str; context: str; holds: bool; how: Literal["proved", "observed"]   # legacy, read as Finding
class Negative:   subject: str; condition: str; effect: str                                    # legacy, read as Finding
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
                 bounds: dict[str, float]               # schema 2 (D464): L1 pair/recurrence bounds in us
                 unpriced: dict[str, float]             # schema 2 (D465): class or "resource:<r>" -> amount L1 could not price
class Build:     ok: bool; regs: int?; spills: int?; smem: int?; log_summary: str
class Check:     passed: bool; failures: list[dict]    # 每项：检查名、输入分布、形状、最大误差等；不含隐藏形状本身
class Measurement: median_us: float; ci_low_us: float; ci_high_us: float; n: int
                 clocks: dict; temps: dict; protocol: str
class Profile:   metrics: dict[str, float]
class Bound:     per_unit_us: dict[Unit, float]; bound_us: float; bottleneck: Unit
class Solution:  bucket: str?; assignment: dict; pct_of_bound: float
class GapDecomposition:                                # WP18 (D520): one per judged round; refs: case, family
                 round: int; gap_class: str            # open: routing.yaml classes or "none"
                 route: dict                           # name, rule, to, task, escalate, improve
                 routing: str                          # "routing@<version>:<sha12>" of orchestrator/routing.yaml
                 buckets: list[dict]                   # per bucket, see below
                 discipline: dict                      # mechanisms, mechanism_count, gain_us, credited,
                                                       # null_round, null_rounds, rolled_back, t_us, band_us, note
```

GapDecomposition bucket item: `name`, `class`, `inputs` (what the decomposition read: `best_us`, `ci_us`, `predicted_us`, `top_pred_us`, `bound_us`, `reference_us`, `pairs`, `solver` {method, model_optimal, exhaustive, mip_gap}, optional `relaxed_pred_us`, `irreducible_us`, `unpriced`, `failed_checks` count), `terms` {`epsilon`, `delta`, `delta_pre`, `l_rep`, `irreducible`: {`us`, `status` ∈ value | bounded | unbounded | not_assessed, `note`}}, `t_meas_us`, `t_sol_us`, `gap_us`, `band_us`, `l_rep_bound_us` (= P1 − T_SOL), `model_off`, `accepted`, `rho`, `median_rel_error`, `selection` {model_best_us, model_top_e_us, random_best_us, random_n, copied_us, copied_from, model_beaten_by_random}. The terms sum to `gap_us` whenever none is `unbounded` (spec/modules/orchestrator-cli.md "WP18").
#### 4.8.1 FormulationPlan, TaskHeader, Pull (WP19; docs/design/02 §5.3, §5.4; D540, D541, D545)

The plan is the structured product of the formulation decisions F1–F7 (D-T18); the
deterministic default is the same object with `author: default` (D-T19). The interface WP18
reads for δ / Δpre attribution is stable: `variables[*].status` (which dimensions were open,
fixed or outer), `chain` and `citations`; `refs.default` names the default plan the plan is
diffed against.

```python
class Citation:      structure: str?          # a Structure name (paper Table 1)
                     evidence: list[str]      # record IDs (Finding, Measurement, Coupling, ...)
class PlanStep:      key: str; template: str; applies_to: str = ""        # F1
class PlanBinding:   item: str; var: str; to: int | str | bool            # F2
class VarStatus:     name: str; status: Literal["open", "fixed", "outer"] = "open"   # F3
                     value: int | str | bool?   # the fixed value
                     menu: list[int | str | bool]?   # F4; None = the declared domain
                     structure: str?; evidence: list[str]                   # its citation
class TableTerm:     effect: str; over: list[str]; by: Literal["compiler", "agent"]   # F5
class GlueSpec:      id: str; expr: str; why: str                           # F6
class DispatchPlan:  runtime_vars: list[str]; buckets: list[dict]           # F6 (default one bucket)
class FormulationPlan:
    family: str                               # "<name>@<version>"
    author: Literal["default", "agent", "human"]
    chain_source: str                         # family | retrieved:<chain id> | enumerated | agent
    chain: list[PlanStep]; bindings: list[PlanBinding]; variables: list[VarStatus]
    table_terms: list[TableTerm]; glue: list[GlueSpec]; dispatch: DispatchPlan
    generator: str?                           # F7: record / blob ID; None = no new generator
    citations: dict[str, Citation]            # chain, bindings, table_terms, glue, dispatch, generator
class TaskHeader:    text: str; situation: str; discipline: str; case: str?; plan: str?
class Pull:          task: str; command: str; target: str; kind: str?
                     situation: str            # the header's situation, "" without a header
                     row: str?                 # the access-discipline row allowing it; None = outside
                     discipline: str           # table version
```
refs: FormulationPlan may have `case` (Case) and `default` (FormulationPlan). TaskCard may have
`plan` (FormulationPlan, the default plan its header shows).


### 4.9 流程与审计

```python
class StatusChange: subject: str; to_status: EvidenceStatus | None; to_assurance: Assurance | None; reason: str
class TaskCard:     type: Literal["T1", "T2", "T3", "T4", "T5", "T6"]; goal: str; inputs: list[str]
                    done_checks: list[str]; budget: dict; profile: str; work_unit: str   # profile：AgentProfile 记录 ID
                    header: TaskHeader?   # schema 2 (WP19): case-loop tasks; TASK.md starts with header.text
class AgentRun:     session: str; profile: str; cli_version: str; model: str
                    tokens_in: int?; tokens_out: int?           # 会话中带有用量时由解析器填写；否则为空
                    session_archive: str                      # blob：该会话的会话文件与 CLI 输出（不含凭据）
                    wall_s: float; tool_s: float; queue_s: float; exit: str
                    context_manifest: str             # blob：本次运行全部上下文来源及其 hash（agents.md）
                    violations: list[dict]            # 越权尝试、未允许的命令、工作区外写入等
                    arm: dict                         # experiment arm: name, knowledge (zero | text | pool), snapshot (D430)
                    proxy: dict?                      # WP22 (D586): used, policy, scheme, host, port, source, no_proxy, preflight; never credentials
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
refs: Report may have `env` (EnvManifest). ControlsVersion has `previous` (ControlsVersion) except version 1. AgentProfile may have `base` (the profile a draft was edited from, WP22 D594). TaskCard and AgentRun may have `controls` (the ControlsVersion the card's budget / the step used), ExperimentRun `controls` (a list: the versions its loop ran under) (WP21). Template's `backward_map_fn: str?` (python function path) is a field of its own.

推进规则（`kta.store.status`）：

| 变更 | 允许的 actor |
| --- | --- |
| 任何对象 → proposed（创建时默认） | 任何 |
| proposed → reproduced / calibrated / holdout_validated | checker |
| 任何 → rejected / deprecated / suspect | human |
| 任何 → stale | system（依赖的记录变更或被驳回时，见 docs/design/20 §6） |
| a promotion level → a lower one (demotion, D448) | system |
| 保证等级 declared → tested / bounded / proved | checker |

不满足规则的 StatusChange 写入时被拒绝。

Interpretation used by the implementation (D149, needs user review): checkers may also promote from `stale` and forward between promotion levels; `-> proposed` is allowed for any actor only while the object is proposed, for a human from any status; assurance only moves up.

Refs of the WP14 kinds: Finding and Coupling `evidence` (records), `sources`; Coupling `family`; CapabilityDecl `target` (Target record, optional), `sources`; Capability `sources`.

**One ladder for empirical knowledge (D446).** Finding, Coupling, Regime and Negative move along proposed → reproduced → calibrated → holdout_validated (02's names observed/claimed, replicated, explained are aliases). Evidence rules (`kta.knowledge.ladder`): reproduced needs ≥ 2 distinct real evidence records; calibrated a resolvable `explained_by` or a Report with `ci_low`/`ci_high`; holdout_validated a Report with `holdout` and `agrees` true. Templates keep assurance and are not on this ladder.

### 4.11 Proof records (WP24; D631–D635)

```python
class Assumption:     name: str; statement: str; expr: str?      # expr over p.* / hw.* / case.* only
class ObligationModel: obligation: str                            # an obligation name of the Template
                       condition: Literal["initialisation", "step", "progress", "completion", "index"]
                       model: str                                 # kta.checks.proof.models.MODELS (open table)
                       params: dict; note: str?
class ProofOutline:   template: str                               # "<name>@<version>"
                      version: int = 1; relation: str; rank: str  # R and ρ (paper §4.1)
                      assumes: list[Assumption]; guarantees: list[str]
                      models: list[ObligationModel]; note: str?
class Query:          sha256: str; result: str; seconds: float    # unsat | sat | unknown | enumerated
class ProofResult:    subject: str; scope: Literal["template", "chain", "candidate"]
                      condition: Literal["K1", "K2", "K3", "encoding", "composition", "candidate", "wp"]
                      obligation: str; outcome: Literal["proved", "bounded", "counterexample", "unresolved"]
                      domain: str; premises: dict[str, str]; premise_key: str
                      queries: list[Query]; seconds: float
                      counterexample: dict?; replay: dict?; reason: str; data: dict
class ProofSummary:   subject: str; scope: ...; part: str = ""    # "step:<key>" for a chain step
                      level: Literal["proved", "bounded"] | None
                      checked: list[dict]; unchecked: list[dict]; gaps: list[str]
                      premise_key: str; checker: str; solver: str
```
refs: ProofOutline `template` (required), `sources`, `previous`; ProofResult `subject` (required), `outline`, `reused`; ProofSummary `subject` (required), `results`, `steps` (chain step summaries).

- `counterexample` is set only after the independent interpreter (`replay`) confirms the witness; an unconfirmed witness, a timeout, `unknown` or an unsupported construct is `unresolved`, never a proof.
- Assurance `bounded` / `proved` of a Template, Family or Chain is read only from its newest live whole-subject ProofSummary (`kta.store.assurance.effective`, D634).
