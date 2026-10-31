# models、formulation、solve

设计依据：docs/design/22 §3.3、50 §6–§7；论文第 5.2 节。

## models（第 4 层）

### 文件

```
models/bound.py        理论下界（per-resource SOL on the hardware model, D464）
models/unit_model.py   按单元的成本模型（编译前、编译后）; the calibration layer over L1 (D466)
models/calibrate.py    校准
models/fit.py          拟合算法（D244）
models/machine.py      MachineModel: HwModel (instance) resolved on its HwArch (class) (D463)
models/l1.py           L1: steady-state pricing of evaluated effects (D464, D465, D471)
models/stored.py       load_arch / load_machine / latest_machine from the store
```

### Hardware model and L1 (WP15; docs/design/02 §4, §5.1)

The hardware model is declarative data (spec/schemas.md §4.6.1): an architecture class `HwArch` and a chip instance `HwModel` whose every value has a `Provenance`. `MachineModel.build` is the only place where it becomes numbers. Templates declare `effects` (§4.2.1); the formulation binds them; L1 prices them. The same three objects are the interface of L2 (WP16) and of the projection compiler (WP17):

```python
class MachineModel:                                    # frozen; models/machine.py
    target: str; hw_version: int; arch: str | None
    hw: Mapping[str, Any]                              # the hw.* values (= HwModel.limits)
    resources: Mapping[str, Resource]                  # rate (chip-level amount/s at peak, None = unmodelled), latency_s, ncu, status, source
    op_classes: Mapping[str, OpClass]                  # rows: (resource, amount Expr), latency_s, serial
    walls: tuple[Wall, ...]; tables; efficiency; interactions; clock_hz; scope_counts
    @staticmethod
    def build(hw: HwModel, arch: HwArch | None = None) -> MachineModel   # arch None: schema-1 flat model
    def op_class(kind: str) -> OpClass | None          # "unit:<u>" is the built-in legacy class
    def rate(resource) -> float | None; def time_s(resource, amount) -> float | None
    def evaluate(expr, tx=None, f=None) -> float       # resolves tbl.* first
    def efficiency_of(resource, features) -> float     # declared efficiency, 1 when none
    version: str                                       # identity of everything that changes a price

# models/l1.py: evaluated effects (numbers, no expressions) — what L2 consumes too
EvalTx(source, id, pattern, kind, count, amounts);  EvalStorage(source, id, level, scope, bytes)
EvalOverlap(source, between, holds, inflight, trips);  EvalEffects(txs, storage, overlaps)

def price_parts(effects, mm) -> (Parts, unpriced, unmodelled, per_tx)   # peak-time vectors, eff = 1
class Parts: busy_peak_us; serial_peak_us; pairs_peak_us; recurrence_us
def compose(parts, eff=None, interactions=(), serial_coef=1.0, serial_const_us=0.0) -> Composed
    # total = max(max_r busy_r / eff_r, pair bounds, recurrences) + serial + sum gamma * min(busy_a, busy_b)
def check_walls(storage, mm) -> tuple[WallCheck, ...]                   # no_launch -> infeasible; lower_occupancy -> residency
def price(effects, mm, features=None) -> L1Result                       # parts + composed + walls + unpriced + unmodelled
class L1Result: feasible; complete; total_us; residency; prediction() -> Prediction
def sol_us(min_work, mm) -> dict[resource, us]                           # per-resource speed of light
```

- **L1 composition** (rule `steady_state_max`): every transaction reserves `count * amount` on each resource of its class's reservation table; `busy_r = use_r / (rate_r * eff_r)` over overlappable transactions; a two-selector overlap that does not hold adds the pair bound `t(G_a) + t(G_b)`; one that holds with `inflight` and `trips` adds `trips * (lat(G_a) + lat(G_b)) / inflight` with `lat(G)` the smallest class latency in the group (optimistic, D464); serial classes (and one-selector overlaps that do not hold) add their time. Total = max of the bounds + serial + interactions. Each term is a lower bound of the steady state; the projection is these terms.
- **For WP16 (L2):** L2 consumes `EvalEffects` (per CTA, before aggregation by `p.instances` when it needs it) and `MachineModel` (resources, op classes, latencies, walls, tables); it must reproduce `price_parts` as its steady-state limit on loop-dominated fixtures (WP16 acceptance). Its outputs use `Resource.ncu` names.
- **For WP17 (MIP):** the compiler reads `Formulation.transactions/storage/overlaps` (expressions over family variables) and `MachineModel`; `formulation.work_terms(f, mm)` already gives every resource-use row as an expression (count × reservation amount, `tx.*` substituted; table terms raise NotSupported). Pair and recurrence bounds and walls are WP17's additions (epigraph max, `II >= lat/S` menus, integer ceil); today's `solve_mip` relaxes them and re-prices its shortlist with L1.
- **Unpriced** (D465): unknown classes and resources without a peak cost no time and are reported in `L1Result.unpriced/unmodelled` and `Prediction.unpriced`; `solve` refuses a formulation with priced effects on a flat (class-less) HwModel (NotSupported).
- `bound(case, shape, hw, work, arch=None)` is the per-resource SOL (`sol_us`); a resource of the minimum work without a peak raises NotSupported. `unit_time_us(unit, work, hw, arch=None)`; `hw` may be an HwModel or a MachineModel everywhere in models, formulation and solve.
- `UnitModel` is the calibration layer (D466): `total_us` delegates to `compose`; `predict_l1(result, mm, feedback=None)` uses its fitted `eff_u` where it has one and the hardware model's declared efficiency elsewhere, and raises `NoResidentCta` on a no-launch wall. `Prediction.bounds` are taken as fixed by refits.

### 接口

```python
def bound(case: Case, shape: dict, hw: HwModel, work: dict[Unit, float]) -> Bound
    # work 为语义上必需的最少工作量（每个输入读一次、每个输出写一次的字节数；语义 FLOPs）；
    # 由 Problem 的参考定义计算，与任何族无关
class UnitModel:
    version: str
    def predict(self, per_unit_work: dict[Unit, float], serial: dict[Unit, float], hw: HwModel,
                feedback: dict | None = None) -> Prediction
        # (D240) serial is the serial work per unit; (D241) Prediction.per_unit / serial are
        # the times at peak (eff = 1) in us, total is this model's prediction in us
        # time_u = work_u / (peak_u * eff_u)；total = max_u time_u + serial + Σ 交互项
        # feedback（编译后）：寄存器、spill、smem → occupancy，修正 eff_u
    def fit(self, data: list[tuple[Prediction, Measurement]]) -> "UnitModel"
        # 拟合 eff_u、serial 系数和交互项；输出新版本，旧版本保留
```

### 行为

- 单元名取自 Target 的单元表。`eff_u` 初值为 1，由校准得到。
- 交互项只在数据支持时加入：两个单元的组合残差显著（按 measure 的显著性规则）时，加入这两个单元的交互项，记录依据。
- 模型的每个版本是一条记录，带拟合数据的 ID 列表和留出点上的误差。

## formulation（第 5 层）

### 文件

```
formulation/assemble.py   从族拼装 formulation
formulation/blocks.py     单个链项的块：变量、约束、成本项（经绑定改名）
```

### 接口

```python
class Formulation:
    variables: dict[str, VarSpec]          # 族变量
    constraints: list[BoundExpr]           # 各链项约束（经绑定）＋ glue；每条带来源（链项 key 或 "glue"）
    cost_terms: list[BoundCost]            # 各链项成本项（经绑定）；每条带来源
    linear: bool                           # 全部约束都属线性子集
def assemble(family: FamilyPackage, chain: Chain, templates: dict[str, Template], case: Case, hw: dict,
             previous: Formulation | None = None) -> Formulation      # previous: reuse unchanged blocks
def per_unit_work(f: Formulation, assignment: dict, shape: dict, hw: dict | MachineModel) -> tuple[dict[Unit, float], dict[Unit, float]]
    # 同一单元的工作量相加；serial 项单独相加，按单元分开（D240）
    # (D460) legacy unit:<u> terms need only hw.*; other transactions need a MachineModel
# (D460, D464) formulation/pricing.py:
Formulation.transactions: list[BoundTx]; .storage: list[BoundStorage]; .overlaps: list[BoundOverlap]
Formulation.cost_terms                                 # only the legacy unit:<u> transactions (schema-1 cost)
def evaluate_effects(f, assignment, shape, hw: dict) -> EvalEffects
def price_point(f, assignment, shape, mm: MachineModel, features=None) -> L1Result
def work_terms(f, mm) -> list[BoundCost]               # MIP rows: count x reservation amount per resource
def cost_breakdown(f, assignment, shape, hw | MachineModel) -> list[dict]   # legacy terms + per-transaction times
    # Binding semantics of chain items, including p.* and the reserved p.instances: D246, D247
```

### 行为

- 绑定：链项中 Template 变量名替换为族变量名或常量。一个 Template 变量绑定到常量时，它从搜索中移除。
- 每条约束和成本项保留来源，用于诊断时说明"是哪一步的约束排除了这个点""哪一步的成本项偏差最大"。
- 换掉链中的一步，只重新生成它那一块，其余不变（增量复用）。

## solve（第 6 层）

### 文件

```
solve/enumerate.py   有限域枚举，带约束剪枝
solve/mip.py         线性子集导出到 HiGHS
solve/select.py      短名单选择（编译前）与重排（编译后）
```

### 接口

```python
def solve(f: Formulation, case: Case, shape_bucket: dict, model: UnitModel, hw: HwModel | MachineModel,
          shortlist: int = 64, time_limit_s: float = 60,
          rng: random.Random | None = None, max_nodes: int = 2_000_000,
          arch: HwArch | None = None) -> SolveResult
    # (D464) points are priced by L1 (model.predict_l1); points on a no-launch wall are dropped
    # and counted in notes; priced effects on a class-less HwModel raise NotSupported
    # (D248) time_limit_s bounds the MIP; enumeration is bounded by max_nodes; rng shuffles
    # the value order of a budgeted (non-exhaustive) enumeration
    # SolveResult：candidates（按预测排序，每个带 Prediction）、方法（enumerate/mip）、
    # 可行点数、剪枝数、是否穷尽、MIP 的 gap 与终止状态
def rerank(candidates: list[Candidate], builds: dict[str, Build], model: UnitModel, hw: HwModel, k: int = 8) -> list[Candidate]
    # builds keyed by Candidate.key (D250)
def record_solve(ctx, store, result, actor, case_id=None, family_id=None, hw_model_id=None,
                 model_id=None) -> SolveRecords      # Candidate, Prediction and a "solve" Report (D251)
```

### 行为

- 变量的笛卡尔积小于 10⁶ 时用枚举，并返回"已穷尽"；否则 `linear=True` 时用 MIP，否则用带剪枝的枚举加随机采样，并标明"未穷尽"。
- 求解结果写 Prediction 记录。"模型内最优"只在已穷尽或 MIP gap 为 0 时才可以声称，并写明适用范围：这个族、这个 formulation、这个模型版本。
- 每个形状桶分别求解；分发规则由族的 DispatchSpec 给出。

### Projection compiler (WP17 inside WP19; docs/design/02 §5.1, §5.2; D546, D547, D548)

`solve/projection.py` compiles a formulation's bound effects, a FormulationPlan's fixed
values / menus / outer variables and the `MachineModel` into the MIP of L1's projection at
peak (eff = 1, interaction terms dropped: a lower bound of L1). The agent writes none of it.

```python
def compile_and_solve(f, shape, mm, *, fixed=None, menus=None, outer=(), shortlist=16,
                      time_limit_s=60.0, table_max=TABLE_MAX, outer_max=OUTER_MAX) -> ProjectionOutcome
    # assignments (objective order across outer points), objective_us, status, first_optimal,
    # outer, outer_points, menus (sizes), rules (uses per rule), notes (why each variable went outward)
def menus_of(f, shape, hw, fixed=None, menus=None) -> dict[var, list]   # F4; ranges > MENU_MAX narrowed by their constraints
def outer_vars(f, mm, shape, menus, outer=None, table_max=TABLE_MAX) -> (set[var], reasons)
def atoms(f, mm, shape, menus, fixed) -> list[Atom]   # rule, row, source, support, value(...), linear?
def projection_us(f, assignment, shape, mm) -> float  # compose(price_parts(...)) at eff = 1
def consistency_check(...)                            # MIP <-> L2 hook: NotSupported until WP16 merges
```

- Rules: ResMII `T >= busy_r` per resource (epigraph max); pair bound `T >= u_a + u_b`,
  `u_g >= t_g,r`; RecMII `T >= trips (lat_a + lat_b) / inflight`; serial time in the
  objective (constants as offset); walls as linear rows over storage atoms (lower_occupancy:
  `bytes + reserved <= limit`, residency >= 1); constraints as linear rows, else as no-good
  rows over their tabulated infeasible combinations.
- Encoding: one-hot menus; an atom over one variable is linear in its menu; over k >= 2
  variables a joint table with linked indicators (menu product, table term; `tbl.*`,
  `min`, `ceil` evaluated in the table); a combination where an atom is undefined (e.g.
  `ceil_div` by zero: no CTA fits) is excluded; an atom over more than `table_max` entries
  takes the linear path when its expression is linear up to `ceil` (integer column
  `n >= L`), else its largest-menu variable moves outward. A two-selector overlap whose
  truth depends on a variable and a `kind:` selector over a variable class move their
  variables outward. More than `outer_max` outer points: NotExportable.
- Translation correctness: at every returned point the objective equals `projection_us`
  (test). Knobs: `formulation.table_max_entries`, `formulation.outer_max_points`.
- `solve(..., projection={...})` takes this arm (method `projection`; `{}` = the default
  plan's inputs, or `plan_inputs(plan, f)`); the shortlist is re-priced by L1 as before. The
  previous MIP export (`solve/mip.py`, hand / agent-written formulation) stays the default
  of `solve` and is labelled the comparison arm of D-T1 (note "formulation arm: ...").
- MIP <-> L2 check (pending WP16): sample the subspace, evaluate `projection_us` and L2,
  require Spearman rho >= threshold and MIP <= L2 on at least a share of points; a failure
  names the effect (atom source) with the largest deviation. Thresholds become registry
  knobs with `set_by: [X11]` when the check is added.

### FormulationPlan and chains (WP19; docs/design/02 §5.3, §5.5; D541–D543)

```python
# solve/plan.py
def default_plan(fp, f, mm, shape, chain_source="family") -> FormulationPlan   # D541
def plan_inputs(plan, f) -> {"fixed", "menus", "outer"}                          # for compile_and_solve
def diff_plans(default, plan) -> list[PlanChange]     # section, key, default, plan, cited
def score_plan(plan, default, f, mm, shape, require_citation=True, solve=True) -> PlanScore
    # changes, uncited, complete (02 §5.4 stop rule), open_dims, space vs default_space,
    # bound_us vs default_bound_us (projection optimum, same chain only)
def record_plan(store, plan, actor, at, case=None, default=None) -> record ID
# solve/chains.py
def start_symbols(templates) -> list[(pattern, level)]
def enumerate_chains(store, case, start, mm, sol_us, incumbent_us=None, max_steps, max_chains,
                     target=None) -> ChainEnumeration   # chains, counts, rules, missing, filtered, best, coverage()
def composition_stats(chains, templates) -> dict; def composition_view(store) -> dict
```

- Grammar (D542): a symbol is (pattern, level); a Template rewrites `(refines, level_from)`
  into `(p, level_to)` for each produced pattern. A legal chain refines every start symbol
  at least once; produced symbols may stay unrefined; newest usable version per name;
  capability filter on the target, Γ false drops a step (unknown kept and counted),
  `excludes`/`requires` relations. Bound: max SOL of the minimal work over the resources the
  chain's operation classes reach (a minimal-work resource not reached: unbounded, counted);
  bound > incumbent: pruned. Knobs `formulation.chain_max_steps`, `chain_max_expansions`.
- Composition statistics are a derived view (co-occurrence, order, conditional frequency
  "pattern@level -> Template") over Chain records; `GET /composition`.

## 验收

- 单元测试：拼装后约束的来源正确；绑定常量后变量被移除；枚举与 MIP 在同一个小线性问题上给出相同最优点。
- 真实运行：WP7 的 GEMM 族在三个形状上求解，前 8 个候选实测，报告预测与实测的排序相关系数（不设通过阈值，作为基线记录）。
