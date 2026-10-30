# 20 存储与检索

## 1. 存什么：三类东西分开存

| 类别 | 例子 | 特性 | 存放方式 |
| --- | --- | --- | --- |
| 证据记录 | Case、Template 版本、Build、Check、Measurement、Gap、状态变更 | 不可修改；彼此引用；构成依赖图 | SQLite 的 `records` 表，只追加 |
| 产物 | 源码、二进制、ncu 报告、agent 事件流、日志 | 可能很大；按内容去重 | 数据盘上的内容寻址目录 |
| 运行状态 | 任务队列、租约、worker 健康、当前池指针 | 需要频繁修改；不作为证据 | SQLite 的运维表，允许修改 |

分开的理由：证据必须可信、可追溯，所以不可修改；运行状态必须能改。混在一起时，"能改"会悄悄扩散到证据上。

## 2. 证据记录

### 2.1 结构

每条记录是一个 JSON 对象：

```json
{
  "kind": "Measurement",
  "schema": 1,
  "created_at": "2026-11-02T10:31:07Z",
  "actor": {"type": "tool", "name": "kta measure", "task": "<task_id>"},
  "refs": {"build": "<id>", "protocol": "<id>", "target": "<id>"},
  "body": { "...": "该 kind 的具体字段" },
  "blobs": {"raw_samples": "<sha256>"}
}
```

- **ID** = 整个对象规范化序列化后的 sha256。ID 就是完整性校验：内容一旦被改，ID 就对不上。
- `refs` 指向其他记录，所有记录由此构成一张有向无环图。
- `actor` 说明这条记录由谁产生：工具、检查器、agent（标明任务）、还是人。
- 数据库触发器禁止对 `records` 表执行 UPDATE 和 DELETE。

### 2.2 状态由事件推导

记录本身不可改，所以"状态"不作为记录里的字段。状态的变化写成 `StatusChange` 记录：

```
StatusChange{subject: <id>, from: "proposed", to: "reproduced", actor: checker, evidence: [<check_id>, ...]}
```

当前状态 = 该对象最新一条有效的 `StatusChange`。"有效"指 actor 有权做这次变更：

- agent 只能创建"提议"；
- 检查器可以推进到"已复现""已校准""留出集验证"；
- 人可以驳回、废弃，或标记为可疑。

### 2.3 记录种类

| kind | 含义 | 主要引用 | 产生者 |
| --- | --- | --- | --- |
| Target | 硬件目标身份：GPU、驱动、CUDA、编译器、容器镜像 | — | 探测工具 |
| HwModel | 一个版本的硬件参数集，带不确定度 | Target、microbenchmark 结果 | 校准工具 |
| HwFeature | 一项硬件能力或一个指令族，及各 Target 是否支持 | Target、Source | 导入脚本或人 |
| HwFact | 关于某项特性或某个单元的参数与行为 | HwFeature、HwModel、证据 | microbenchmark 或导入脚本 |
| Problem | 计算语义、参考实现、dtype、布局 | — | 人或实验配置 |
| Contract | 正确性合同和计时边界 | Problem | 人 |
| Case | Problem＋Contract＋形状分布＋目标函数 | Problem、Contract、Target | 人或实验配置 |
| Source | 冻结的外部来源：commit 或 hash、license | — | `kta` 导入 |
| Reference | 参考实现：来源和构建方式 | Source、Problem | 人或任务 T2 |
| Pattern | 事务模式及其参数 | — | 人或 worker（你确认） |
| Template | 细化步骤的一个版本（22 §2） | 上一版本、refines 的 Pattern、requires 的 HwFeature、Source | 任务 T3、T6 |
| Chain | 组合链 | 各链项的 Template 版本 | 任务 T1、T2 |
| Family | 可执行族的一个版本：链＋生成器 | 上一版本、Chain、产物（代码） | 任务 T1、T6 |
| Structure | 可利用的结构（论文表 3） | — | 人 |
| Regime | 结构在某个情境下成立或失效的证据 | Structure、证据 | 求解或校准 |
| Candidate | 一组 assignment | Template、Case | 求解器 |
| Prediction | 模型对候选的预测 | Candidate、HwModel、模型版本 | 求解器 |
| Build | 编译结果和编译反馈 | Candidate、Target | 执行器 |
| Check | 检查结论 | Build、Contract | 检查器 |
| Measurement | 实测结果 | Build、测量协议、Target | 执行器 |
| Profile | profile 指标 | Build | 执行器 |
| Bound | 理论下界 | Case、HwModel | 工具 |
| Reconstruction | 参考实现拆成的组合链、匹配率、未匹配事务、复现误差 | Reference、Chain、Measurement | 任务 T2 加检查 |
| Gap | 缺口：缺什么、证据、归因类别 | 触发它的记录 | 循环或 agent |
| Solution | 某个 case、某个形状桶的最佳赋值 | Case、Template、Candidate、Measurement | 求解循环 |
| Negative | 某个做法在什么条件下无效或有害 | 所针对的 Template 或 Family、证据 | 消融、求解循环或 worker |
| TaskCard | 任务定义 | 循环状态、输入记录 | 编排器或人 |
| AgentRun | 一次 agent 运行的摘要 | TaskCard、产物（事件流） | 运行器 |
| Submission | agent 的交付物和交付说明 | AgentRun | agent（经 `kta submit`） |
| StatusChange | 状态变化 | 对象记录、证据 | 检查器或人 |
| HumanAction | 人的介入操作及理由 | 被操作的对象 | 人（经前端或 CLI） |
| PoolSnapshot | 一个池快照 | 上一快照、成员记录 | 编排器 |
| Protocol | 测量协议或实验协议 | — | 人批准 |
| ExperimentRun | 一次实验运行 | Protocol、PoolSnapshot、Case | 编排器 |
| Report | 改进报告或实验报告 | 汇总的记录 | 工具 |

## 3. 产物存储

- 位置：控制面数据目录下的 `objects/ab/cdef…`，按 sha256 分目录存放。worker 上只有缓存，以控制面为准。
- 写入时先计算 hash，内容相同则只存一份。
- 记录通过 `blobs` 字段引用产物，`blobs` 表保存大小、类型和首次写入时间。
- 大文件（ncu 报告、事件流）默认长期保留。二进制可以重新编译，但也保留，以免工具链变化后无法复现。

## 4. 运行状态

运维表允许修改，不作为证据：

| 表 | 内容 |
| --- | --- |
| `tasks` | 任务状态、优先级、租约持有者、租约到期时间、重试次数 |
| `loops` | 每个循环实例的当前状态 |
| `workers` | worker 的健康状态和心跳 |
| `locks` | GPU 安静租约 |
| `pointers` | 当前池快照等指针 |
| `ops_events` | 上述各表每次变化的流水，只追加，供前端显示和排查 |

任务被接受、被驳回这类结论，会同时写成证据记录（StatusChange、AgentRun）。所以运维表即使丢失，证据也不会丢失。

## 5. 池与快照：怎样表示不同的知识库

- **池快照**是一条 PoolSnapshot 记录，列出它包含的 Pattern、Template、Family 版本、Reference、硬件事实和校准数据集，并指向上一个快照。
- 新知识被接受后（状态推进到"已复现"或以上），编排器生成新快照，并更新 `pointers.current_pool`。
- 每次求解和每次实验运行都**固定引用一个快照**，结果因此可以复现。
- 实验中的不同知识条件：

| 条件 | 怎样构造 |
| --- | --- |
| 从零构造 | 空快照 |
| 文字知识 | 用确定性的渲染器，把某个快照渲染成 Markdown 知识书，给 agent 阅读；不提供可执行 Template |
| Template 池 | 直接使用该快照 |
| 不同的积累程度 | 在加入第 0、n、2n… 个参考实现后，各保存一个快照 |

文字知识条件和 Template 条件使用同一份知识，区别只在形式。这样比较才公平。

- 换一种知识库构造方式（例如只从某类来源学习），就是从另一个起点生成另一条快照线。快照之间互不影响。

## 6. 版本与失效

- 依赖图由 `refs` 构成。`edges` 表把 `refs` 展开，方便查询。知识对象之间的边（uses、implements、fits 等）也存在这张表里，类型见 [23 §3](23-knowledge-structure.md)。
- 当以下对象有新版本、或被驳回时，按依赖图向下标记过期：
  - Target 变化（驱动、工具链升级）；
  - HwModel 有新版本；
  - Contract 变化；
  - 检查器版本变化；
  - 某个 Template 版本被驳回。
- "过期"本身也是一条 StatusChange（actor 为系统）。
- 过期标记会生成重新验证任务。默认只自动重验锚点集（已复现的参考实现，以及实验结果表引用的点），其余等到用到时再重验。

## 7. 检索

KTA 的检索需求有五类，各用最简单的可靠方式：

| 需求 | 例子 | 方式 |
| --- | --- | --- |
| 找适用的 Template | 新 Case 来了，哪些 Template 能用 | ① 按 manifest 的结构化字段过滤（算子类型、dtype、架构、布局）；② 在 Case 的各形状桶上执行合法条件代码；③ 排序：在相似 Case 上的历史结果、复用成功率 |
| 找相关知识 | agent 想知道"decode 的 split 有哪些做法" | 按标签过滤（算子、瓶颈单元、硬件特性），再用 SQLite FTS5 全文检索，按 BM25 排序 |
| 找相似的历史 Case | 给求解器提供热启动 | Case 特征向量（算子、dtype、形状统计、下界瓶颈单元、算术强度）上的最近邻 |
| 追证据链 | 这个结果为什么可信 | 在依赖图上遍历（`kta why`） |
| 取校准数据 | 拟合成本模型 | SQL：按 Target、HwModel 版本、Template 版本，连接 Measurement 和 Prediction |

**为什么暂时不用向量数据库：** Template 的适用性是精确条件，结构化过滤和合法条件代码能给出可验证的答案。模糊的知识检索用全文检索先顶上。只有当记录显示"agent 找不到其实存在的知识"反复发生时，才加入向量检索。

**agent 怎样检索：**

1. `kta query` 只返回摘要列表：ID、标题、一句话说明、状态，每次最多 10 条。
2. agent 需要细节时，再用 `kta show <id>` 取完整内容。

这样，agent 的上下文里只有它真正用到的东西。每次检索都被记录，所以能事后检查"agent 有没有看到关键知识"。

## 8. 位置、备份与 git

- 数据库和产物都在控制面所在机器的数据目录（`$KTA_DATA/<env>/`）。开发阶段在本机；正式运行时放在哪台常驻机器，由配置决定。
- 每天用 SQLite 的备份 API 做一次数据库快照，连同新增产物一起 rsync 到 slai。
- git 里只放代码、文档、配置、提示模板、skills。数据不进 git，避免触发大规模打包。
- Template 以存储为准。`kta export` 可以把某个快照导出成目录，方便浏览。

## 9. 规模

第一阶段估计：记录数在百万条以内，产物在几十到几百 GB（主要是 ncu 报告和事件流）。SQLite 足以应对。

需要换存储的信号：并发写入成为瓶颈，或者需要多台机器同时写。届时把 Store 接口换一个实现，不影响上层模块。

## 10. Store 接口

上层模块只通过这个接口访问存储：

```
put_record(record) -> id
get_record(id) -> record
query(kind, filters, order, limit) -> [record]
search_text(query, kind, filters, limit) -> [摘要]
lineage(id, direction, depth) -> 子图
status(id) -> 当前状态
put_blob(bytes | path) -> sha256
get_blob(sha256) -> path
snapshot(members, parent) -> pool_id
```

运维表另有一个 `Ops` 接口：任务的入队、领取、续约、完成，以及锁和指针。

## 依据与待核实

- 正式运行时控制面放在哪台机器、数据目录的容量和备份目标：待定。
- 规模估计是推断，没有依据。
- 结构化检索加全文检索够不够用，需要在 M2 之后用记录检验。
