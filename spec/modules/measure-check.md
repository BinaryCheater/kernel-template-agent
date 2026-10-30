# measure 与 checks

设计依据：docs/design/50 §2–§5。

## measure

### 文件

```
measure/protocol.py     Protocol 定义（一个 Protocol 记录对应一份冻结的协议）
measure/runner.py       在 gpu-exclusive 池中执行测量
measure/profile.py      Nsight Compute 采集与解析
measure/inputs.py       输入生成器（与 checks 共用，见下）
```

### 接口

```python
def measure(ctx: Context, module: BuiltModule, case: Case, shape: dict, protocol_id: str) -> Measurement
def compare(ctx: Context, a: BuiltModule, b: BuiltModule, case: Case, shape: dict, protocol_id: str) -> Comparison
    # 两者按随机顺序交错执行；返回各自的 Measurement 与差值的置信区间、是否显著
def profile(ctx: Context, module: BuiltModule, case: Case, shape: dict, metrics: list[str] | None = None) -> Profile
```

### 默认协议（Protocol 记录 `measure-v1`，数值待 WP2 实测噪声后修订，修订即出新版本）

| 项 | 取值 |
| --- | --- |
| 计时 | CUDA events，每次迭代单独计时 |
| L2 冲刷 | 每次迭代前写一块 2×L2 容量的缓冲区 |
| 输入 | 至少 4 组，轮换 |
| 输出 | 每次运行前填 NaN |
| 预热 | max(10 次, 50 ms) |
| 正式 | max(100 次, 200 ms) |
| 统计 | 中位数；bootstrap 95% 置信区间（1000 次重采样） |
| 环境 | 记录开始和结束时的 SM 时钟、显存时钟、温度、功耗；能锁频则锁频 |
| 显著性 | 置信区间不重叠，且差值 > 3 × 噪声底 |
| 内存 | cudaMalloc |

### 要求

- 测量只在 `gpu-exclusive` 池中执行。
- 测量结果中的 `protocol` 字段写协议记录 ID。不同协议的结果不能放在同一张比较表里，报告生成器检查这一点。

## checks

### 文件

```
checks/numeric.py       数值检查
checks/sanitizer.py     compute-sanitizer
checks/anticheat.py     反作弊检查
checks/sass.py          SASS 统计与预期特征比对
checks/space.py         族的空间验证
checks/hidden.py        隐藏形状集（私有种子）
checks/inputs.py        输入分布（随机正态、全零、放大极端值、按算子的对抗输入）
```

### 接口

```python
def check_numeric(ctx, module: BuiltModule, problem: Problem, contract: Contract, shapes: list[dict]) -> Check
def check_sanitizer(ctx, module: BuiltModule, problem: Problem, shape: dict) -> Check
def check_anticheat(ctx, module: BuiltModule, gk: GeneratedKernel, problem: Problem, shape: dict) -> Check
def check_sass(module: BuiltModule, expectations: list[SassExpect], assignment: dict) -> Check
def full_check(ctx, module, gk, problem, contract, case) -> Check      # 以上全部，含隐藏形状
def verify_space(ctx, items: Iterable[SpaceItem], subjects: list[str]) -> SpaceReport
    # SpaceItem: path_id、assignment、GeneratedKernel、BuiltModule、problem、contract、kind("path" | "sample")
    # kind="path" 的项做 full_check（边界形状＋隐藏形状）；kind="sample" 的项做 check_numeric；
    # 全部通过 → 对 subjects（族及其链中的全部 Template）写 StatusChange(to_assurance="tested", actor=checker)
    # 枚举路径、生成和编译由 orchestrator 完成（checks 不 import knowledge）：
    # 每条可达的结构路径一个 "path" 项，另随机抽 4 个合法 assignment 作为 "sample" 项
```

### 行为

- **数值检查**：参考结果在 GPU 上用 FP64（大形状可用 FP32，记录原因）。判据：逐元素 `|y − ref| ≤ atol + rtol·|ref|`，另检查整体相对 L2 误差。Template 或族的 `numeric_risk` 列出的输入分布必须包含在内。
- **隐藏形状**：由 `checks/hidden.py` 用存放在 agent 不可读位置的私有种子生成。`Check.failures` 中只报告失败的检查名、输入分布、误差，不报告隐藏形状的具体取值，只给出它所属的类别（例如"非 tile 整数倍的长度"）。
- **反作弊**：
  - 输出缓冲区预填 NaN，运行后确认全部被覆盖；
  - 用 CUPTI 或 nsys 追踪，确认只启动了声明的 kernel；
  - 检查模块的动态符号，确认没有链接 cuBLAS、cuDNN、PyTorch 等库；
  - 同一输入重复运行，输出一致（除非 Contract 声明允许不确定性）；
  - Contract 和 baseline 的 hash 与 Case 引用的一致。
- **SASS 检查**：统计指令种类；对每条 `SassExpect`，在 `when` 条件成立时检查 `must_contain` 和 `must_not_contain`。不一致时 `Check.passed = False`，失败类型为 `intent_not_realized`，并写一条 Gap（attribution="intent_sass"）。
- 检查器自身出错（异常、超时）时，返回的 Check 标记 `error_kind="checker_error"`，调用方据此挂起任务，不计为 agent 的失败。

## 验收

- 真实运行：对一个手写的正确 kernel 和三个故意出错的变体（越界写、漏写一行输出、偷调 cuBLAS），检查器分别给出通过和三种对应的失败。
- 噪声测量：同一 kernel 连续测量 30 组，报告中位数的变异系数，作为噪声底写入交付说明，并据此修订 `measure-v1` 的数值（出 `measure-v2`）。
- 交错比较：两个已知性能相差约 5% 的 kernel，`compare` 判定显著；两个相同 kernel 判定不显著。

## WP3 implementation notes (D220–D234)

Changes and additions to the interfaces above, made in WP3 and recorded in docs/decisions.md:

- Return values carry the record ID: `measure` → `Measured(id, body, times_us)`, `compare` → `Comparison(a, b, rel_diff, ci_low, ci_high, significant, verdict, report_id)` (the comparison is a Report record), every check → `Checked(id, body)`. A checker error is a failure item with `error_kind: "checker_error"` (D224).
- `full_check(ctx, module, gk, problem, contract, case, *, case_id, shape, build_for, sass=None, assignment=None)`; `build_for(shape) -> (GeneratedKernel, BuiltModule)` compiles edge and hidden shapes (D227). `SpaceItem` adds `case`, `case_id`, `shape`, `build_for`, `sass`. `check_sass(module, expectations, assignment, *, ctx)` (D229).
- `Measurer` and `Checker` hold explicit store, executor, clock and rng for callers with their own instances; the spec functions resolve them from the Context.
- `Problem.reference_impl` is a torch statement run in the worker image in FP64 (D223). Distributions: `normal`, `zeros`, `uniform`, `extreme` (measure), plus `cancellation`, `dominant_tail` (checks).
- Kernel tracing uses nsys (D221); modules may be `cubin` or `so` (D222); `profile` raises NotSupported where performance counters are admin-only (D228).
- Protocol records `measure-v1` and `measure-v2` (D226); the measured noise floor is in deliveries/WP3.md.
