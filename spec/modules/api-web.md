# api 与 web

设计依据：docs/design/30。

## api（第 9 层）

### 文件

```
api/app.py          FastAPI 应用；持有 Store 的唯一写连接
api/read.py         读接口
api/actions.py      操作接口（写 HumanAction）
api/events.py       SSE
api/auth.py         任务令牌（agent 的 kta 命令经 API 写入时使用）
```

### 接口

读：

```
GET /records/{id}                       记录，含当前状态与保证等级
GET /records?kind=&where=&limit=        查询
GET /search?q=&kinds=                   全文检索
GET /lineage/{id}?depth=                证据链
GET /cases/{id}/timeline                求解循环的每一轮、每个任务、每次驳回
GET /families/{id}/chain                组合链及每步的卡片
GET /coverage?snapshot=                 CoverageReport
GET /model/residuals?model=             预测对实测
GET /runs/active                        正在运行的循环、任务、作业、租约
GET /budget                             各层预算
GET /inbox                              待处理事项
GET /events?since=                      SSE：ops_events、新记录、span 的开始与结束
GET /trace?from=&to=&lanes=&trace=&kinds=   时间窗内的 span（含进行中的）；可按泳道、trace、种类过滤
GET /trace/summary?from=&to=            各泳道利用率；各种类耗时合计；排队占比
GET /trace/critical-path/{trace_id}     一个循环实例或实验运行的关键路径
```

操作（每个都写 HumanAction，`reason` 对驳回、废弃、调整预算为必填）：

```
POST /actions/pause        {scope}
POST /actions/resume       {scope}
POST /actions/cancel       {task}
POST /actions/retry        {task, note}
POST /actions/budget       {scope, limit, reason}
POST /actions/priority     {task, priority}
POST /actions/create-task  {type, goal, inputs}
POST /actions/reject       {record, reason}
POST /actions/flag         {record, reason}
POST /actions/approve      {protocol | improvement}
POST /actions/confirm      {pattern | structure}       新增的模式或结构需你确认
```

写入：`POST /write/record`（只接受带有效任务令牌或本机工具身份的请求）。

### 行为

- 只监听 spark 的本机端口；通过 SSH 隧道访问。不做登录。
- 所有写入经过同一个写连接，保证 SQLite 单写。

## web

TypeScript、React、Vite、ECharts。独立目录 `web/`。

### 页面（按 docs/design/30 §1）

第一版（WP9a）只读：运行中、**流水图**、Case（含时间线）、结果详情（证据链）、族（组合链与每步卡片）、覆盖。

第二版（WP9b）：待处理、任务详情（实时工具调用）、池、模型、实验、改进，以及全部操作按钮。

### 要求

- 每个数字和结论都能点开对应记录。
- 所有图表可以切换池快照。
- 覆盖页顶部始终显示抽样范围（来源数量与类别）。
- 流水图（docs/design/30 §1.2）：
  - 横轴为时间，纵轴为泳道；每个 span 一个色块，颜色按种类；
  - 支持缩放、拖动、"跟随当前时间"的实时模式（经 SSE 更新）；
  - 点击任务或会话色块，展开其内部的模型段与工具调用段；
  - 悬停显示名称、起止、耗时、状态、关键 attrs；点击跳到对应记录；
  - 可按 Case、实验运行、进程、主机过滤；
  - 顶部显示时间窗内各泳道的利用率，以及 GPU 空闲比例。
- 组合链用有向图展示：节点是链项（显示 Template 名、细化的事务、保证等级），边是 `after` 依赖。

## 验收

- 真实运行：在本机通过 SSH 隧道打开前端，看到 WP7 的一次真实求解的时间线和结果证据链；截图写入交付说明。
- 每个操作接口有测试：写出的 HumanAction 字段正确；缺少必填理由时返回 400。
