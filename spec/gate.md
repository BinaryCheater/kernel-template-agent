# 验收脚本 gate

`scripts/gate <WP>` 是每个 WP 结束前必须运行并通过的检查。它的作用是防止"以为做完了"：mock 或 toy 实现被当成完成、测试被削弱、能力没有真实运行过。

## 输出

逐项打印 `PASS`、`FAIL` 或 `WARN`，以及原因；任何一项 FAIL 时退出码非零。最后打印一段可直接粘贴进交付说明的摘要。

## 通用检查（所有 WP）

| 检查 | 方法 | 失败条件 |
| --- | --- | --- |
| 格式与 lint | `ruff format --check`、`ruff check` | 有错误 |
| 类型 | `pyright`（core、store、orchestrator 严格模式） | 有错误 |
| 依赖方向 | `lint-imports` | 违反层次或禁止项 |
| 单元测试 | `pytest -m "not gpu"` | 有失败 |
| 残留检查 | 在 `src/` 中搜索 `NotImplementedError`、`TODO`、`FIXME`、`XXX`、`mock`、`MagicMock`、`pytest.skip` | 出现且所在行没有 `# kta:allowed <原因>` 注释 |
| fixture 申报 | 在 `src/` 中搜索 `# kta:fixture` | 出现的每一处都必须列在交付说明的"fixture 与简化"中 |
| 验收测试锁定 | `tests/acceptance/LOCK` 中记录的每个文件的 sha256 | 已锁定的文件被修改 |
| 交付说明 | `deliveries/<WP>.md` | 缺少 AGENTS.md 规定的任何一节 |
| 决策日志 | `docs/decisions.md` | 新增行格式不对，或交付说明列出的决策 ID 不存在 |
| 跳过的测试 | pytest 报告中的 skip 和 xfail | 未在交付说明中解释（WARN） |
| 替身位置 | 在 `src/` 中搜索类名或文件名含 Fake、Stub、Dummy、Mock 的定义；检查 `src/` 是否 import `tests` | 出现 |
| 注册表 | 在 `KTA_ENV=dev` 下运行 `kta doctor` | 任何接口选到 fixture 实现 |
| 合同测试 | `pytest -m contract` | 有失败；或 `tests/fakes/` 中某个替身没有对应的合同测试 |
| 错误处理 | AST 扫描 `src/`：裸 `except:`；`except Exception` 后只有 `pass` 或 `return` 常量；函数体只有 `pass` | 出现且没有 `# kta:allowed` |
| print | AST 扫描 `src/` 中的 `print(`，`cli` 模块的输出函数除外 | 出现 |
| 机密 | 正则扫描仓库改动中的常见 key 格式 | 命中 |
| 证据清单 | `deliveries/<WP>.evidence.json` 的格式；远端核对每个 record_id 存在于该 WP 的 dev 存储、origin 为 real、git_commit 匹配、dirty 为 false | 任一不满足（需要真实运行的 WP） |

## WP 专项检查

| WP | 额外检查 |
| --- | --- |
| WP0 | 每个模块目录有 README.md；import-linter 的违例样例测试存在并通过 |
| WP1、WP2、WP3 | 运行 `pytest -m gpu tests/acceptance/<WP>/`，GPU 作业经 Executor 在 spark worker 上执行；存储中存在本 WP 期间创建的、origin="real" 的对应记录（WP1：合成数据测试的计时记录；WP2：Target、Build；WP3：Measurement、Check、Protocol） |
| WP4、WP5 | seed 全部通过校验；`pytest tests/acceptance/<WP>/` |
| WP6 | 状态转移函数的每条判定规则都有测试 |
| WP7 | 存在一次 `solve_loop` 的完整记录，最终状态为 `accepted` 或有 judgement 记录；三种故障注入各有一条失败的 Check 记录；重放脚本输出一致 |
| WP8 | 存在至少一条 AgentRun（origin="real"），以及一次会话恢复的记录 |
| WP9a、WP9b | 前端构建通过；API 测试通过；交付说明中附截图 |
| WP10a | `knowledge/seed/references/` 下每个参考实现都有 Reference、Chain、Reconstruction；YAML 能被模型解析；未匹配事务与 Gap 一一对应 |
| WP10b | Named WP10b tests pass (expand machine, WP10a replay frozen → covered via Python and CLI, cancelled extraction, inflated `matched`); expand transitions create only T1–T3 and reach `covered`; T2/T3 done-checks exist. Full stage: ≥ 10 reference directories and a listed clean coverage Report in `dev/WP10b` with a covered reference (D388) |
| WP13 | 交付说明中的每个数字都有来源文件路径 |
| WP14 | Named WP14 tests pass (old-format records read back, unprobed Template filtered on GB10 and counted, promotion/demotion with stale propagation, pool diff, seed migration idempotent); the seed loads under the new schema with Capabilities, GB10 declarations and Realisations; the seed migration `--check` reports nothing pending; no `requires_hw` left in seed Template YAML; every 02 §2 object kind is registered; the delivery note has the design-interrogation artefacts and a pool-diff screenshot. Full stage: a probe-backed CapabilityDecl and a non-empty `probed_on` from a real run in `dev/WP14` |
| WP20 | Named WP20 tests pass (the seven WP10a decompositions admitted with counted language gaps, an unexplained warning refuses the batch, the language-gap table, tiers/priority/performance core, ablation design and coupling evidence, fractional design, dry-run plan, a T2 rejected by admission and resumed in the same fake-adapter session); the seven references import through admit into a scratch store with language gaps counted and the coverage page showing four SOTA levels and a system-feature count; the learning table has tiers L0–L4 with known rules, levels 1–4, the O12 protocol marked not run, and T2 runs the `admission` done-check; the delivery note has the design-interrogation artefacts and pending real runs. Full stage: real intake, reproduce and ablate runs on spark and an O12 run |

## 验收测试的锁定

- 每个 WP 在 `tests/acceptance/<WP>/` 中，把对应规范"验收"一节写成测试。
- WP 合入 main 时，由合并者把这些文件的 sha256 追加到 `tests/acceptance/LOCK`。
- 之后的 WP 不得修改已锁定的文件。确实需要修改时，在交付说明的"需要的变更"中提出，由用户决定。

## 强制运行

AGENTS.md 规定最后一步必须运行 gate 并把输出贴进交付说明。合入前再由合并者运行一次。如果某种 agent 工具支持"结束前钩子"，可以把 `scripts/gate --quick <当前 WP>` 配进去作为加强，但不依赖它。

## Code stage (D320)

`scripts/gate --stage code <WP>` is the code-complete gate used while the user has paused real data and measurements (2026-10-31). It runs every check except the ones that need real runs: evidence claims need no store records, and WP-specific checks that need a GPU run, a measurement or real records return WARN through `specific.pending_real(...)` when `specific.code_stage()` is true. The delivery note must add a non-empty section `## Pending real runs` listing what the full gate will still require. A WP that passed only the code stage may be merged into `integration` as code-complete; it is not accepted until `scripts/gate <WP>` (full) passes.
