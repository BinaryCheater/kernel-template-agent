# 23 知识的结构

KTA 的知识是一张**有类型的图**：节点是知识对象，边是关系。每个节点都是存储中的一条记录（20），带来源、证据状态和保证等级。知识以结构化字段存储，可以检索、关联、组合；每个节点另附简短的文字说明和最小代码示例，供 agent 和人理解。

## 1. 全图

```
 问题侧                     设计侧                                   硬件侧
 Problem/Case ─has─► Pattern ◄─refines── Template ──requires──► HwFeature ◄─about── HwFact
                     (事务模式)            (细化步骤)                (硬件能力)          (参数与行为)
                                          │  ▲  relations：requires / excludes /
                                          │  │  enables / refines / alternative_of
                                          │  └───────── Template
                               step_of    ▼
 Reference ─reconstructed_as─► Chain ◄─has_chain── Family ◄─assigns── Solution
 (外部实现)                    (组合链)              (可执行族)          (case→赋值＋实测)

 Structure（可利用的结构）──held_in──► Regime 证据（在什么 workload、什么芯片上成立）
 Negative ──about──► Template / Family 区域（在什么条件下无效或有害）
 Gap ──missing──► 缺的是什么（模式、Template、关系、硬件事实）
 所有节点 ──source──► Source ；所有节点 ──status──► StatusChange
```

## 2. 节点

| 节点 | 是什么 | 关键内容 | 是否依赖硬件 |
| --- | --- | --- | --- |
| **Pattern** | 一种事务模式，带参数 | 所在层；种类（reduce、contract、map、store、transfer、sync、access、other）；参数及其性质 | 否 |
| **Template** | 带契约的细化步骤（22 §2） | 见 22 §2.1 | 合法性和成本按 Target 求值 |
| **Chain** | 一串 Template 实例及绑定 | 链项；分支；作用对象 | 是（经由 Template） |
| **Family** | 可执行族：链＋生成器 | 生成器代码；代码标记；胶合约束；分发；边界形状 | 是 |
| **Solution** | 某个 case、某个形状桶的最佳赋值 | Family 版本；赋值；实测；%下界 | 是 |
| **Reference** | 外部参考实现 | 来源；构建方式；实测；重建出的链；未匹配的部分 | 是 |
| **HwFeature** | 一项硬件能力或指令族 | 所属单元；相关指令；各 Target 是否支持 | 是 |
| **HwFact** | 参数或行为 | 数值；不确定度；文档还是实测 | 是 |
| **Structure** | 可利用的结构（论文表 3）：瓶颈主导、近可分解、单调性、对称性等 | 它对搜索做什么（压缩目标、压缩变量、压缩枚举）；需要的证据 | 否 |
| **Regime** | 某个结构在某个情境下成立或失效的证据 | workload、芯片、证据、是被证明还是只被观察到 | 是 |
| **Negative** | 某个做法在什么条件下无效或有害 | 对象；条件；证据 | 通常是 |
| **Gap** | 缺了什么 | 缺失类型；触发它的证据；归因 | 视情况 |
| **Source** | 冻结的外部来源 | commit 或 hash；license | — |

## 3. 边

| 边 | 从 → 到 | 含义 |
| --- | --- | --- |
| has | Case → Pattern | case 的计算包含哪些模式 |
| refines | Template → Pattern | Template 细化哪种模式 |
| produces | Template → Pattern | 细化后产生哪些模式 |
| requires | Template → HwFeature | 需要哪些硬件能力 |
| about | HwFact → HwFeature | 事实描述哪项能力 |
| requires / excludes / enables / refines / alternative_of | Template → Template | 组合规则 |
| step_of | Template → Chain | Template 作为链中的一步 |
| has_chain | Family → Chain | 族的设计 |
| reconstructed_as | Reference → Chain | 参考实现被表达成哪条链 |
| assigns | Solution → Family | 解是哪个族的哪组赋值 |
| derived_from | Family → Family | 族的父版本 |
| held_in | Structure → Regime | 结构在哪些情境下成立 |
| about | Negative → Template 或 Family | 负面结论针对什么 |
| missing | Gap → 节点类型 | 缺什么 |
| source | 任意 → Source | 出处 |

## 4. 各功能怎样在图上计算

| 功能 | 怎样计算 |
| --- | --- |
| 检索适用的 Template | case 的 Pattern → refines 它们的 Template → 过滤 Γ、requires |
| 按诊断检索 | 瓶颈单元 → 涉及该单元的 Pattern（transfer、store、access 等）→ 细化它们的 Template |
| 差集检索 | 相似 Reference 的 Chain 中的 Template，减去当前 Chain 中的 Template |
| 组合检查 | 链中各 Template 之间的 requires、excludes 是否满足 |
| 增量复用 | 最近的 Chain → 逐步求值 Γ（22 §4） |
| 覆盖统计 | Reference 的链匹配率；Pattern 中 other 的数量；Template 的保证等级分布；HwFeature × Pattern 网格（21） |
| 失效传播 | HwFact 变化 → about 的 HwFeature → requires 它的 Template → step_of 的 Chain → Family → Solution |
| 负面知识提醒 | 检索到某个 Template 时，带出 about 它的 Negative |
| 文字知识条件 | 按 Pattern 遍历子图，渲染成 Markdown（20 §5） |

## 5. 思路、代码、硬件分开

- **Template 是思路加契约，基本不依赖具体硬件。** 硬件以 requires 的能力和 Γ 中的硬件契约出现。
- **Family 是代码，依赖硬件。** 同一条链，在不同 GPU 上可能需要不同的生成器。
- **HwFact 是测量和文档，完全依赖硬件。**

换硬件时，Template 层基本保留，只需重新求值 Γ 和成本；Family 层需要检查能否编译运行；HwFact 层整体重测。

## 6. 谁能创建什么

| 节点 | 谁可以提议 | 状态怎样推进 |
| --- | --- | --- |
| Pattern | worker 或人；新增后需你确认，因为它影响检索和覆盖统计的口径 | 你确认 |
| Template | worker | 保证等级见 22 §6；证据状态见 21 §1 |
| Chain、Family | worker | 族通过空间验证 |
| Reference、Source | 导入脚本或人 | 在我们的协议下通过检查并完成实测 |
| HwFeature | 导入脚本（从 PTX ISA 文档整理）或人 | 探测确认 |
| HwFact | microbenchmark 或导入脚本 | 重复测量一致 |
| Structure | 人或 worker；初始取论文表 3 | 你确认 |
| Regime、Negative | 求解与消融的结果 | 有实测或证明 |
| Gap | 循环或 worker | 解决后关联到补上的节点 |

## 依据与待核实

- 图的规模在第一阶段为几千个节点，存于 SQLite 的 records 和 edges 表即可。
- 模式表、HwFeature 清单的初始内容需要在 WP 中整理，覆盖性通过 21 的统计和 24 的重建持续检验。
