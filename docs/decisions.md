# 决策日志

每条一行。状态：**需要你**＝等你决定；**默认**＝助手先选、施工可以按它进行、尚未确认；**已确认**＝你确认过。新决策加在对应分组末尾，不删除旧条目；改变决策时，新增一行并注明替代哪一条。

## 需要你决定

| ID | 决策 | 暂用默认 | 文档 |
| --- | --- | --- | --- |
| D01 | 主结论 | 摊销、覆盖为主；饱和、预测、质量、可诊断为辅 | 00 §1 |
| D02 | 强模型和弱模型的具体型号 | 无，实验开始前必须确定 | 50 §1.2 |
| D03 | 总预算：周期、token、GPU 时间、你的审阅时间 | M0 按小额预算执行 | — |
| D04 | 闭源库（如 cuBLAS）能否做 SASS 分析 | 暂不分析，只用开源参考实现 | 20 §2.3 |
| D05 | 隔离形式：本仓库的顶层目录，还是独立仓库（已由 D66 决定） | 本仓库顶层目录 | README |
| D06 | 前端页面划分和待处理的触发阈值 | 按 30 中所写 | 30 |
| D07 | 有人工改动的运行怎样计入 | 单独标记，不进主表 | 50 §1.4 |
| D09 | agent 的隔离程度：只用配置文件与环境变量隔离（config）；再加独立系统用户（config+user）；或容器（container） | config+user | spec/modules/agents.md |

## 默认（未确认）

| ID | 决策 | 选择 | 文档 |
| --- | --- | --- | --- |
| D10 | 流程控制 | 确定性代码驱动流程（由 D26 的独立进程运行各循环）；只在创造性工作上调用 LLM | 10 §1 |
| D11 | agent 权限 | agent 只能提议；只有检查器和人能推进状态；合同、检查器、隐藏集、baseline 对 agent 不可写 | 10 §4.7 |
| D12 | agent 隔离（隔离程度改由 D09 决定） | 以独立 unix 用户运行，没有 GPU 设备权限；GPU 只能经执行器使用 | 10 §4.7 |
| D13 | provider 接入（已被 D65 替代） | 适配器接口；第一批为 Claude Code 和 Codex；需支持会话恢复，不支持的每步开新会话 | 10 §4.6 |
| D14 | 证据存储 | SQLite 只追加；记录 ID 为内容 hash；状态由 StatusChange 推导 | 20 §2 |
| D15 | 产物存储 | spark 数据盘上的内容寻址目录；每天备份到 slai | 20 §3、§8 |
| D16 | 知识库变体 | 用池快照表示；文字知识由同一快照渲染 | 20 §5 |
| D17 | 检索方式 | 结构化过滤＋合法条件代码＋FTS5；暂不用向量库 | 20 §7 |
| D18 | 部署位置（已被 D70 替代） | 控制面、存储、agent、编译都在 spark；正式测量时持有安静租约；slai 用于备份和可选的额外算力 | 11 §4–5、20 §8 |
| D19 | 模块与依赖方向 | 按 40 §2 划分；用 import-linter 强制 | 40 |
| D20 | 软件栈 | 按 40 §6 | 40 |
| D21 | 测量协议、正确性检查、空间验证 | 按 50 §2–4；数值待 M0 校准 | 50 |
| D22 | 性能模型 | 按单元瓶颈模型，编译前和编译后两阶段；目标是排序 | 50 §7 |
| D23 | 施工方式 | 主 agent＋skills；骨架先行；接口冻结后并行 | 60 §1 |
| D24 | 防假完成 | done gate＋独立验收（不同模型）＋里程碑验收 | 60 §4–6 |
| D25 | 第一阶段范围 | GB10，单 kernel 加小融合；不做 graph 和 engine 层 | 00 §9 |
| D26 | 进程划分 | 求解、学习、校准、维护、实验、汇总六个独立进程，只经存储和队列通信 | 11 §1 |
| D27 | 预算层级 | 全局 → 进程配额 → 循环实例 → 任务 → 作业；先预留后结算；配额默认实验 50%、求解 25%、学习 15%、校准与维护 10% | 11 §3 |
| D28 | agent 预算的计时 | 排队时间不计入；总墙钟时间硬上限为预算的 3 倍 | 11 §3.3 |
| D29 | 你的注意力作为资源 | 待处理事项超过 10 项时，暂停会产生升级的新任务 | 11 §2 |
| D30 | 锁 | GPU 独占租约、循环实例锁、Template 线写锁、快照指针 CAS、单写连接；按固定顺序取锁 | 11 §5 |
| D31 | 工具、skills、池知识三分 | skill 只写方法，不写领域知识；检索动作必须经过工具留下记录 | 10 §6 |
| D32 | agent 角色划分 | 只按权限边界、独立判断和并行划分；运行时只有 worker（另有可选的只读审计）；不按任务类型划分 | 10 §4.1 |
| D33 | 会话粒度 | 一个工作单元一个会话，多步之间恢复会话；状态写在 NOTES.md；超过上下文阈值、重复失败或单元结束时换新会话；实验每次运行用新会话 | 10 §4.2 |
| D34 | 上下文策略 | 固定前缀（便于缓存）＋一页步骤头＋按需获取；工具输出默认精简；重复工作由编排器批量完成 | 10 §4.4 |
| D35 | 实例数 | 默认 2，上限 4，按预算、速率限制、积压和 GPU 排队调整；同一工作单元内不并行 | 10 §4.5 |
| D36 | 知识准入 | 记录门槛低（带来源即可记为提议）；按状态分级使用；负面知识同样入池；入池前查重 | 21 §1 |
| D37 | 覆盖估计 | 发现曲线与 Chao1/Good–Turing、硬件指令族与特性网格、结果侧差距解释率与输入空间热力图，三个方向同时展示 | 21 §2、30 §1.1 |
| D38 | 学习优先级 | 学习待办按覆盖缺口排序 | 21 §3 |
| D39 | 知识分层（已被 D49 替代） | 事实、机制、组件多积累；骨架和 Template 少而通用；新 Template 须证明不能表达为已有 Template 的赋值或特化 | 22 §1 |
| D40 | Template 格式（已被 D49 替代） | 论文格式为核心，外加签名、槽位、分发、数值风险、SASS 特征、耦合、边界形状、特化；卡片、性能图、状态、复用统计只由工具生成 | 22 §2 |
| D41 | Template 检索（已被 D58 替代） | 编排器确定性检索：硬过滤 → 按证据排序 → 最多 5 张卡片；机制按诊断键和差集检索；每个算子族一页概览 | 22 §3 |
| D42 | 组合（已被 D51 替代） | 只在声明了接口的槽位上自动组合；跨层融合经 prologue/epilogue 槽位的语义类型；第一阶段只做 kernel 内和小范围融合 | 22 §4 |
| D43 | 知识的结构（已被 D59 替代） | 有类型的图：HwFeature、HwFact、OpFamily、Mechanism、Component、Skeleton、Template、Reference、Solution、Negative、Gap、Source；思路、代码、硬件事实分层 | 23 |
| D44 | 新增算子族（已被 D50 替代） | 需你确认，因为影响检索和覆盖统计的口径 | 23 §6 |
| D45 | kernel 设计的六个方面（已被 D49 替代） | 计算、分解与映射、数据移动与驻留、计算指令、调度与重叠、同步与资源；每个机制标明所属方面 | 24 §1 |
| D46 | 原型（已被 D49 替代） | 跨算子复用的执行结构（一组经过验证的相容机制）作为独立节点；骨架＝原型×计算变体 | 24 §2 |
| D47 | 机制之间的关系（现适用于 Template 之间，见 D49） | requires、excludes、enables、refines、alternative_of；新机制入池必须声明与已有机制的关系 | 24 §3 |
| D48 | Template 边界（已被 D49 替代） | 一个 Template ＝ 一个计算变体族 × 一个原型；同一代码可参数切换的做成变量，需要不同代码结构的分成兄弟 Template | 24 §4 |
| D49 | 知识单元与 kernel family | 知识单元＝论文定义的细化 Template：细化一种模式的粗事务、引入一个对象，带 X、C、Pre、P、π、Σ、Ĵ、Γ 及检索与组合所需字段；kernel family＝组合链＋生成器（可执行族）；不预设设计"方面"或"原型" | 22 §1–§3 |
| D50 | 事务模式表 | 开放的数据表，初始 8 类；描述不了的记为 other 并计数；Γ 用模式参数写，知识按结构关联而不按 kernel 名；新增模式需你确认 | 22 §2.2、23 §6 |
| D51 | 组合方式 | formulation 由链项的变量、约束、成本块拼装；代码层面的组合先由 agent 写生成器并加代码标记，按 Template 出现频率逐步做成自动变换 | 22 §3 |
| D52 | 增量复用 | 新 case 从最近的组合链出发，逐步求值 Γ，只处理不成立的步骤；摊销指标为链项复用比例 | 22 §4 |
| D53 | 检查义务后移 | 保证等级 declared → tested → bounded → proved；第一阶段只要求 tested（空间验证）；形式化保证在迭代中补上，前端显示分布 | 22 §6、21 |
| D54 | 覆盖的检验方式 | 持续把参考实现拆成组合链；统计匹配率、未匹配事务、other 数量、保证等级分布 | 21 §2.4、24 |
| D55 | 代码生成后端 | 先用 TileLang 与 CUDA C++；CuTe DSL、MLIR 在 sm_121 探测后再评估 | 22 §3.4 |
| D56 | 施工规范的组织 | AGENTS.md＋spec/schemas.md＋spec/modules/＋spec/workplan.md＋spec/gate.md；工作包可由不同 agent 并行；WP10a（手工拆解）最早开始，用于检验知识模型 | 60 §2–§3 |
| D57 | 施工的 git 流程 | 每个工作包一个分支 wp/<编号>；不 push、不合入 main，由用户决定 | AGENTS.md |
| D58 | 检索 | 族与 Template 由编排器检索：模式匹配 → Γ → 证据排序 → 卡片；按诊断：瓶颈单元 → 模式 → Template；差集；族概览 | 22 §5 |
| D59 | 知识图的节点 | Pattern、Template、Chain、Family、Solution、Reference、HwFeature、HwFact、Structure、Regime、Negative、Gap、Source | 23 |
| D60 | 记录、产物、日志的分工 | 结论只能来自记录（及其引用的产物）；日志只用于调试，保留 30 天，JSON lines，带任务与作业 ID | spec/conventions §1、§4 |
| D61 | 环境隔离 | KTA_ENV＝test/dev/main；main 只接受真实实现、origin=real 和已提交的代码；实现经注册表选择，dev/main 下选到 fixture 实现即拒绝启动 | spec/conventions §2–§3 |
| D62 | 防止假实现 | 替身只在 tests/fakes 且必须通过合同测试；做不到抛 NotSupported；证据清单逐条远端核对；gate 用 AST 扫描 src 中的替身、吞异常、print、空函数 | spec/conventions §9–§10、spec/gate |
| D63 | 远端运行方式（已被 D70、D71 替代） | 本地提交后用 scripts/sync 同步到 spark 的 WP 目录，uv sync --frozen 后运行；不在 spark 上改代码；有未提交修改的运行不能作证据 | spec/conventions §5–§6 |
| D64 | 机密 | API key 只放在 spark 的 ~/.config/kta/secrets.env，由运行器注入；gate 扫描常见 key 格式 | spec/conventions §8 |
| D77 | Implementation registry form | default — needs user review: the registry is an object on `ctx.registry` (no module-level table); packages expose `register_implementations(registry)`; selection = `[impl]` name, else the single real implementation; spec/conventions.md §3.2 updated | WP0 |
| D78 | Subprocesses outside the Executor | `kta.core.envprobe` may run fixed, read-only version commands (git, uv, node, nvcc, nvidia-smi) because doctor and EnvManifest run before any Executor exists and core cannot import execution; the gate rejects subprocess use anywhere else outside `kta.execution` | WP0 |
| D79 | Where the gate lives | `tools/kta_gate/` (construction tooling, not in the `kta` package), run by `scripts/gate` through `uv run --frozen`; tests import it via pytest `pythonpath` | WP0 |
| D80 | Node for pyright | dev dependency `pyright[nodejs]`: pyright runs on the Node runtime from the `nodejs-wheel` in `.venv`, with caches under `.venv`; no global Node and no user-level nodeenv | WP0 |
| D81 | Extra dev dependency | `types-pyyaml` so pyright can check code that uses PyYAML | WP0 |
| D82 | Delivery note headings | canonical English `##` headings (What was done, Fixtures and simplifications, Known issues, New decisions, Needed changes, Gate output); the Chinese names in AGENTS.md are accepted as aliases | WP0 |
| D83 | Residue scan scope | only `.py` files under `src/` are scanned for TODO/FIXME/mock etc.; Markdown skills and prompts under `src/` are not | WP0 |
| D84 | Evidence verification | default — needs user review: record-backed evidence is required for WP1, WP2, WP3, WP7, WP8, WP9a, WP9b, WP10b, WP11, WP12; the gate reads `$KTA_DATA/dev/<WP>/kta.db` and requires `actor.commit` and `actor.dirty` in the record JSON (WP1 must name the Actor fields this way); other WPs may list claims with empty `record_ids` | WP0 |
| D85 | Unimplemented WP-specific gate checks | fail the gate; the WP that owns them adds them to `tools/kta_gate/specific.py` with tests | WP0 |
| D86 | WP13 traceability rule | every line with a measured number (decimal, percent, or time/ratio unit) carries an `[S<n>]` citation defined in a `[S<n>] <path>` sources list; local absolute paths must exist | WP0 |
| D87 | WP10a file classification | a YAML document's kind is its `kind` field, else its file name (`reference*`, `chain*`, `reconstruction*`, `gap*`, or a `gaps/` directory); Gap documents must equal `unmatched_transactions` per reference | WP0 |
| D88 | Data directory variables | `dev` requires `KTA_WP` (data in `$KTA_DATA/dev/<WP>/`); `test` requires `KTA_TEST_DATA`; missing values are a ConfigError, never a silent default | WP0 |
| D89 | Remote project roots | spark `~/kta-work`, slai `/storage/kta-work` (slai's root disk is full); set in `config/remote.toml` | WP0 |
| D90 | Pinned toolchain | Python 3.12.13 (`.python-version`); uv 0.11.x; `scripts/bootstrap` installs a missing interpreter into the project's `.uv/python`, never the user-level uv directory; uv's download cache stays at its default location | WP0 |
| D91 | Config hash | sha256 of the parsed TOML serialized as sorted, compact JSON (formatting and comments do not change it); WP1 may switch to `canonical_json` with the same meaning | WP0 |
| D92 | Control-plane data root during construction | `KTA_DATA=~/kta-data` on this machine (outside both git repos); spark workers use `~/kta-work/data` | WP0 |

## 已确认

| ID | 决策 | 选择 | 来源 |
| --- | --- | --- | --- |
| D08 | 项目名称 | KTA（Kernel Template Agent）；目录 `kta/`，Python 包与命令行为 `kta`。论文中的方法名 Formulaic 保持不变，与本系统分开 | 2026-10-30 用户确认 |
| D65 | 运行时 agent 的 provider | 第一阶段只接 Codex；Claude Code 适配器保留接口，方法抛 NotSupported，暂不运行；worker 固定前缀写在工作区 AGENTS.md | 2026-10-30 用户决定 |
| D66 | KTA 的形态 | 独立的 agent workspace：自己的 git 仓库（位于研究仓库 kta/ 下，研究仓库本地忽略它）、自己的 AGENTS.md 和 .agents/skills/；遵循 AGENTS.md 标准，不用 CLAUDE.md；所有新 skill 放在 workspace 级 | 2026-10-30 用户决定；具体布局为助手默认 |
| D67 | 流水图 | 所有有起止的工作记 span（进程步骤、任务、agent 思考与工具调用、作业排队与运行、锁、人工等待）；前端按泳道展示顺序、并行与耗时，含利用率与关键路径；列入 WP9a 第一版 | 2026-10-30 用户提出；字段与页面为助手默认 |
| D68 | agent 行为的控制 | 上下文、循环策略、工具全部写进有版本的 AgentProfile；每次运行隔离 CLI 配置目录（不读用户级配置、AGENTS.md、skills）、关闭 web search 与 MCP、环境变量白名单；运行前生成上下文清单；切换 profile 需你批准；agent 不能改自己的 profile、提示或 skills；实验固定 profile | 2026-10-30 用户提出；具体内容为助手默认 |
| D69 | 调用 AI 必须可追溯 | 所有 AI 调用经适配器；会话持久保存在工作单元的 agent-home 目录并打包存档；记录 token 与费用；禁止临时会话模式和不留记录的临时环境 | 2026-10-30 用户要求 |
| D70 | 部署拓扑 | 控制面与计算资源可在不同机器和集群，不共享文件系统；计算资源只经 Executor；传输方式 local、ssh、pull（pull 先留接口）；按能力标签选 worker；产物按 hash 传输。开发阶段控制面在本机，GPU worker 为 spark | 2026-10-30 用户要求；具体方案为助手默认 |
| D71 | 运行环境自包含 | Python 用 .python-version＋uv.lock，前端用 .nvmrc＋package-lock.json，只经锁文件安装；每台机器生成 EnvManifest；doctor 对照锁文件；bootstrap 与 worker-setup 脚本可在新机器上重新部署 | 2026-10-30 用户要求；具体方案为助手默认 |
| D72 | 适配器 | 先实现 Codex；Claude Code、pi coding agent、DeepSeek harness 保留接口，接入前按准入要求（非交互、事件流、会话持久、配置可重定向、可关联网检索、可指定 API 地址）逐项核对 | 2026-10-30 用户要求；准入要求为助手默认 |
| D73 | 施工地点 | 在本机开发与运行控制面；施工由用户另开会话进行 | 2026-10-30 用户决定 |
| D74 | 事件的两个层次 | 系统事件（编排、任务、作业、锁、会话起止、kta 调用）由 KTA 实时记录；agent 循环内部的事件不要求实时记录，要求会话完整保存，需要时由会话解析器或 harness 提供的事件得到，流水图中按需展开 | 2026-10-30 用户要求 |
| D75 | 用量与费用 | KTA 不要求记录 token 和费用；会话中带有用量时顺带记下；分析由用户的其他工具完成，必要时再补脚本；预算按运行时间、GPU 时间和步骤数控制 | 2026-10-30 用户要求 |
| D76 | 认证方式 | Codex、Claude Code 用订阅：登录凭据放在受保护目录，按工作单元链接进隔离配置目录；pi、DeepSeek harness 用 API：key 在 secrets.env。凭据不进上下文清单内容、会话存档和日志 | 2026-10-30 用户说明；存放方式为助手默认 |
