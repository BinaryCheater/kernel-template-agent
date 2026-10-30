# models、formulation、solve

设计依据：docs/design/22 §3.3、50 §6–§7；论文第 5.2 节。

## models（第 4 层）

### 文件

```
models/bound.py        理论下界
models/unit_model.py   按单元的成本模型（编译前、编译后）
models/calibrate.py    校准
models/fit.py          拟合算法（D244）
```

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
def per_unit_work(f: Formulation, assignment: dict, shape: dict, hw: dict) -> tuple[dict[Unit, float], dict[Unit, float]]
    # 同一单元的工作量相加；serial 项单独相加，按单元分开（D240）
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
def solve(f: Formulation, case: Case, shape_bucket: dict, model: UnitModel, hw: HwModel,
          shortlist: int = 64, time_limit_s: float = 60,
          rng: random.Random | None = None, max_nodes: int = 2_000_000) -> SolveResult
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

## 验收

- 单元测试：拼装后约束的来源正确；绑定常量后变量被移除；枚举与 MIP 在同一个小线性问题上给出相同最优点。
- 真实运行：WP7 的 GEMM 族在三个形状上求解，前 8 个候选实测，报告预测与实测的排序相关系数（不设通过阈值，作为基线记录）。
