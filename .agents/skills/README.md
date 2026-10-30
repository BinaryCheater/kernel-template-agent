# 施工 skills

本目录存放施工 agent（构建 KTA 本身的 coding agent）使用的 skills，每个 skill 一个子目录，入口为 `SKILL.md`。运行时 worker 的 skills 不在这里，在 `src/kta/agents/skills/`。

| skill | 用途 | 由哪个 WP 编写 |
| --- | --- | --- |
| `run-gate` | 怎样运行 `scripts/gate`、怎样读它的输出、怎样处理失败 | WP0 |
| `write-delivery` | 交付说明和证据清单的写法 | WP0 |
| `remote-run` | 同步代码到 spark、在远端运行、后台运行与查看日志、查询远端 dev 存储 | WP0 写通用部分，WP2 补全数据目录和环境 |
| `design-interrogation` | 设计或修改任何机制、流程、spec 前后必做的一遍追问：九个问题、对抗性二审、三种必须产出（决策台账、待证义务、控制项） | 设计讨论（2026-10-31） |
