# 70 其他 kernel agent：对照清单

目的：迭代 KTA 时，对照其他 kernel agent 的设计，确保我们没有漏掉关键点。我们的判断是，现有做法大多依赖模型能力反复试错，知识不能结构化地积累，所以这里不照搬它们的设计，只用它们来查漏。

## 1. 对照维度

每个系统都按这些维度检查一遍：

| 维度 | 要问的问题 | KTA 的做法 |
| --- | --- | --- |
| 跨任务保留什么 | 什么都不保留、聊天历史、文字笔记、代码，还是结构化的空间和模型 | 可执行 Template、校准过的模型、负面知识（21） |
| 知识怎样检索 | 每次通读、向量检索，还是结构化检索；检索是否留下记录 | 结构化过滤＋全文检索；每次检索都有记录（20 §7） |
| 下一步由谁决定 | LLM 每轮决定，还是求解器 | 求解器在空间内搜索；LLM 只扩展空间（10 §1） |
| profile 怎样用 | 每轮都跑，还是按需 | 只在预测与实测偏差大时跑（50 §7） |
| 正确性与防作弊 | 怎样检查，能不能被绕过 | 隐藏形状、反作弊检查、agent 没有写权限（50 §5） |
| 测量协议 | 是否固定，是否可被 agent 影响 | 固定协议，独占 GPU（50 §2） |
| 每个 kernel 的成本 | 是否报告 token 和时间 | 记录时间；会话完整存档，token 与费用可事后分析（10 §8） |
| 迁移 | 换形状、换硬件时，之前的工作还能用多少 | 换形状：重新求解；换硬件：重新校准（40 §4） |
| 覆盖 | 能否说出自己还缺什么 | 三个方向的覆盖估计（21 §2） |
| 人的角色 | 人在哪里介入 | 待处理队列和介入操作（30） |

## 2. 已知的系统

"了解程度"一栏：**摘要**＝只看过检索摘要；**读过**＝读过论文或文档；**运行过**＝实际跑过代码。目前全部是"摘要"，各条描述在读原文之前都不能当作事实。

| 系统 | 一句话 | 跨任务保留 | 了解程度 |
| --- | --- | --- | --- |
| [Kernel Design Agents](https://www.sourcepulse.org/projects/30472462)（MIT HAN Lab） | Claude Code 工作流：任务合同（目标、约束、验证），每轮做验证、benchmark 和 profile，记录结果 | 结果记录；未见结构化知识 | 摘要 |
| [KDA-Pilot](https://www.besthub.dev/articles/how-sglang-encoded-engineering-experience-into-agents-and-achieved-up-to-2-75-kernel-speedups-c5d370745dff)（SGLang） | 把任务隔离开，agent 不能改整个仓库；报告单个 kernel 1.279×、B200 上请求级 1.125× | 工程经验编码进 agent（形式待查） | 摘要 |
| [Atrex Kernel Agent](https://gitee.com/alibaba/atrex-kernel-agent)（阿里巴巴） | ncu profile → 提取瓶颈证据 → 查参考 → 每轮只做一项优化；支持 Codex 和 Claude Code | 待查 | 摘要 |
| [Agentic Kernel Optimization](https://arxiv.org/pdf/2608.14560) | Planner 读取当前最好的 kernel、NCU 结果和历史运行，提出下一步方向 | 历史运行记录 | 摘要 |
| [AscendOptimizer](https://arxiv.org/pdf/2603.23566) | 在公开 kernel 稀缺的 Ascend 上，agent 从自身执行中积累优化知识 | 优化知识 | 摘要 |
| [AgenticCANN](https://arxiv.org/pdf/2607.26661) | 知识增强的 Ascend C 算子生成；知识注入使 elementwise 算子可行率从 57% 升到 86% | 知识库 | 摘要 |
| [SAGE](https://www.besthub.dev/articles/teaching-ai-to-tune-ascend-npus-fudan-huawei-xiaoyi-achieve-95-5-kernel-generation-success-ec34c711067e)（复旦、华为） | 知识冷启动、工具辅助诊断、经验学习；与专家实现对比；NPUKernelBench 成功率 95.5% | 经验 | 摘要 |
| [AscendCraft](https://arxiv.org/html/2601.22760) | 用分类别的专家示例引导 DSL，再转译为 Ascend C | 专家示例 | 摘要 |

你提到的"把生产 kernel 的技巧逐个去掉、学习每项技巧的作用"的 Ascend 论文，可能是上面 Ascend 相关的某一篇，需要读原文确认。

凭记忆列出、尚未检索的：KernelBench（评测集）、Sakana AI CUDA Engineer（曾被报道钻了评测漏洞）、AlphaEvolve（Google）、CUDA-L1（强化学习）、Kevin（Cognition，多轮强化学习）。

## 3. 目前看到的共同模式

就检索摘要看，多数系统的主循环是：profile → LLM 提出改动 → 实现 → 验证。知识主要以文字或历史记录的形式保留，每次运行都要重新推导。

据此，KTA 要证明的差异是：知识以可执行的空间和模型保留；空间内的搜索由求解器完成；profile 按需使用；覆盖可以度量。这些目前是假设，需要用 00 §1 的主结论来证明。

可以直接借鉴的检查项：

- 任务隔离，agent 不能修改仓库的其他部分（KDA-Pilot）：我们已有（10 §4.7）。
- 每轮只改一项，便于归因（Atrex）：对应我们的消融（21 §1.2）。
- 固定的 benchmark，防止 agent 钻测量的空子：对应 50 §2 与 §5。

## 4. 怎样维护

- 每个里程碑结束时，更新一次本表。优先把与我们差异最大、或声称效果最好的系统，从"摘要"推进到"读过"。
- 发现其他系统做到了而我们没有考虑的点，写进决策日志，标明来源。
