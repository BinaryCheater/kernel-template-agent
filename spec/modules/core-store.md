# core 与 store

设计依据：docs/design/20、23；schema 见 spec/schemas.md。

## core

### 文件

```
core/schemas/        每个 kind 一个 pydantic 模型；registry.py 维护 kind 名 → 模型
core/ids.py          canonical_json, record_id
core/expr.py         表达式解析、自由变量检查、线性判定、求值
core/config.py       读取 TOML 配置；计算配置 hash
core/context.py      Context：store、ops、executor、target、config、clock、rng、env
core/registry.py     接口实现的注册与选择（conventions §3.2）
core/errors.py       错误类型（conventions §3.3）
core/logging.py      JSON lines 日志、contextvars、机密打码（conventions §4）
core/doctor.py       kta doctor：列出各接口选用的实现、环境、数据目录；dev/main 环境下发现 fixture 实现即失败
core/trace.py        span 的开始、结束、父子关系、contextvars 传递（conventions §4.1）
```

### 接口

```python
def canonical_json(obj: Any) -> bytes
def record_id(record: Record) -> str

class Expr:
    @staticmethod
    def parse(src: str, declared: set[str]) -> "Expr"   # 语法或自由变量不合法时抛 ExprError
    free_vars: frozenset[str]
    is_linear: bool
    def eval(self, env: Mapping[str, Any], hw: Mapping[str, Any], p: Mapping[str, Any], case: Mapping[str, Any]) -> Any

def load_config(path: Path) -> Config        # Config.hash 为规范化 TOML 的 sha256
```

### 要求

- 表达式求值不得使用 Python 的 `eval`；用 `ast` 解析后自行解释。
- `record_id` 对同一内容在不同进程、不同机器上结果相同（有单元测试，含中文字符串和浮点）。

## store

### 文件

```
store/sqlite.py      Store 与 Ops 的 SQLite 实现
store/blobs.py       内容寻址产物存储
store/status.py      状态与保证等级推导、推进规则
store/stale.py       失效传播
store/snapshot.py    池快照
store/migrations/    建表 SQL
store/db.py          connection, migrations (PRAGMA user_version), timestamp format
store/querybuild.py  `query` filters (D147); store/textindex.py: FTS fields (D151)
store/sqlite_ops.py  Ops on SQLite; store/transfer.py: verified bundles (D153); store/bench.py: benchmark (D156)
```

### 表

```sql
records(id TEXT PRIMARY KEY, kind TEXT, schema INT, created_at TEXT, actor_type TEXT, actor_name TEXT,
        task TEXT, origin TEXT, json TEXT)                -- 触发器禁止 UPDATE 和 DELETE
edges(src TEXT, rel TEXT, dst TEXT)                        -- 从 refs 展开；知识关系（23 §3）也在这里
blobs(sha256 TEXT PRIMARY KEY, size INT, media TEXT, first_seen TEXT)
records_fts                                                -- FTS5，索引 Template.summary、Pattern.description 等文本字段
-- 运维表（允许修改）：
tasks(id, card_id, card_type, state, priority, lease_owner, lease_until, attempts, work_unit, created_at, updated_at)
loops(id, kind, subject, state, data_json, updated_at)
workers(name, host, healthy, heartbeat)
locks(name, holder, until)
pointers(name, value)
ops_events(seq INTEGER PRIMARY KEY, at, table_name, row_id, change_json)
spans(span_id TEXT PRIMARY KEY, parent_id, trace_id, kind, lane, name, start_ts, end_ts, status, attrs_json)
                                                           -- 索引：(start_ts)、(trace_id)、(lane, start_ts)
budgets(scope, limit_json, used_json, reserved_json)
```

### 接口

```python
class Store:
    def put(self, record: Record) -> str                 # 校验 schema、refs 存在、写 records 与 edges；返回 ID；幂等
    def get(self, id: str) -> Record
    def query(self, kind: str, where: dict | None = None, order: str | None = None, limit: int = 100) -> list[Record]
    def search(self, text: str, kinds: list[str] | None = None, limit: int = 10) -> list[Hit]   # Hit: id, kind, title, snippet, score
    def neighbors(self, id: str, rel: str | None = None, direction: Literal["out", "in"] = "out") -> list[str]
    def lineage(self, id: str, depth: int = 10) -> Graph
    def status(self, id: str) -> tuple[EvidenceStatus, Assurance | None]
    def put_blob(self, data: bytes | Path, media: str) -> str
    def blob_path(self, sha256: str) -> Path

class Ops:
    def enqueue(self, card_id: str, priority: int, work_unit: str) -> str
    def lease(self, worker: str, kinds: list[str], seconds: int) -> Task | None
    def renew(self, task_id: str, worker: str, seconds: int) -> bool
    def finish(self, task_id: str, state: str) -> None
    # Added by WP6 on SqliteOps (D265): lifecycle compare-and-set and listings.
    def update_task(self, task_id: str, expected: Collection[str], state: str,
                    owner: str | None = None, lease_s: int | None = None,
                    count_attempt: bool = False) -> bool
    def set_task_priority(self, task_id: str, priority: int) -> bool   # D367, also on Ops
    def list_tasks(self, work_unit: str | None = None,
                   states: Collection[str] | None = None) -> list[Task]
    def list_loops(self, kind: str | None = None) -> list[dict]
    def acquire(self, lock: str, holder: str, seconds: int) -> bool
    def release(self, lock: str, holder: str) -> None
    def set_pointer(self, name: str, expected: str | None, value: str) -> bool   # compare-and-set
    def reserve(self, scope: str, amount: dict) -> bool
    def settle(self, scope: str, reserved: dict, used: dict) -> None
    def events(self, since: int) -> list[OpsEvent]
```

Additions made in WP1 (D146, D152): `Store.put_many(records)` (one transaction), `exists`, `get_json`, `kind_of`, `count`, `blob_media`; `Ops.get_pointer`, `set_budget`, `budget`, `task`, `heartbeat`, `put_loop`/`get_loop`, `start_span`/`end_span` (the `kta.core.trace.SpanSink`), `spans`, `prune_spans`. `query(where=...)` syntax: D147. Additions made in WP2 (D192): `Ops.worker_health(name)`, `Ops.record_event(table, row_id, change)`; `heartbeat` is part of the Ops protocol; `kta.core.trace.current_tracer()`; `kta.core.jobs` holds `Job`, `JobResult`, `JobRunner` (D180). `stale.propagate(store, id, reason, now)`, `stale.reject(...)`, `snapshot(store, ops, members, parent, note, actor, now)` take the store explicitly.

### 行为

- **环境检查**：`Store` 打开时读取 `ctx.env`。`main` 环境拒绝 origin 为 fixture 或 simulated 的记录；任何环境下，actor 中都写入代码的 git commit 和 dirty 标志，`main` 环境拒绝 dirty 的写入。
- 所有写入经过同一个进程内的一个写连接（API 服务持有；CLI 和其他进程通过 API 写，见 api-web.md）。单元测试中可以直接用 Store。
- `put` 对已存在的 ID 直接返回，不报错。
- 状态推导：取该对象所有 StatusChange 中 actor 合规的最新一条（规则见 schemas.md §4.9）。
- 失效传播：某条记录被驳回、或 Target、HwModel、Contract、检查器版本有新记录时，`stale.propagate(id)` 沿 edges 的入边找依赖它的记录，写 StatusChange(to_status="stale", actor=system)，并返回需要重验的记录 ID 列表。
- 快照：`snapshot(members, parent, note)` 写 PoolSnapshot 并用 CAS 更新指针 `current_pool`。

### 验收

- 单元测试：`main` 环境拒绝 fixture 记录和 dirty 写入；注册表在 dev 环境拒绝 fixture 实现；ID 稳定性；UPDATE/DELETE 被触发器拒绝；refs 缺失时 put 失败；状态推进规则逐条测试；失效传播在一个五层依赖图上正确；CAS 指针并发更新只有一个成功。
- 真实运行：在 spark 的数据目录上建库，写入 10 万条合成记录（标 origin="fixture"），查询和全文检索 p95 < 50 ms，记录耗时写入交付说明。

## Controls registry (WP21, D480, D481)

Every knob of docs/design/12-agent-controls.md lives in one versioned registry. Code reads knob values only through it; no module keeps its own default for a knob.

### Files

```
core/controls.yaml     knob definitions: id, row (the "Knob" cell of 12), section, controls, type,
                       value (initial), prevents, where, status, source, set_by, editable,
                       accessor, consumer, unit, choices / fields / min / max / sum_max
core/experiments.yaml  X1–X9 (02 §11): question, design, arms, metrics, sets (knob ids),
                       profile_fields, depends_on, status (defined | approved | running | done)
core/controls.py       definitions(), coerce(), Controls (snapshot), baseline(), upgrade()
core/knobs.py          Knobs: one typed property per knob (mixed into Controls)
store/controls.py      ControlsVersion records and the pointer controls:current
```

### Interface

```python
# kta.core.controls (layer 0: every module may read it)
definitions() -> Definitions          # knobs, experiments, task_types, hash; validated on load
class Controls(Knobs):                # frozen snapshot
    values: Mapping[str, Any]; version: int; version_id: str | None; definitions_hash: str
    def value(knob_id) -> Any                          # a copy
    def with_changes(changes, allow_locked=False) -> Controls   # validated proposal, version_id None
    def diff(other) -> list[Change]
    <accessor>: typed property per knob (e.g. rounds_per_case_loop -> int, task_wall_s -> dict)
baseline() -> Controls                # the definitions' initial values, not stored
upgrade(stored_values) -> (values, added, dropped)

Context.controls: Controls            # snapshot read by pure code; default baseline()

# kta.store.controls (layer 1)
current(store, ops) -> Controls       # read only; version_id None when nothing (or not all) stored
ensure(store, ops, at) -> Controls    # writes version 1, or a system upgrade when knobs were added/dropped
propose(store, ops, changes, author, reason, at, expected) -> Controls
history(store, limit) -> list[(id, Record)]
```

### Rules

- A value change is a new ControlsVersion (spec/schemas.md §4.10) written by a named human with a reason; `expected` must be the version the author read (None = the values in force were not stored yet), else ConfigError. Invariant knobs (`editable: false`) are refused (PermissionDenied). Invalid values are ConfigError.
- `value` in `controls.yaml` only seeds version 1 and knobs added later; editing it does not change a stored value.
- Runs record the version they used: `refs.controls` on TaskCard, AgentRun (one version) and ExperimentRun (the list the loop ran under); loop data `controls` (last step) and `controls_seen`.
- Entry points that hold a store call `ensure` (Services.open, the runner per step, the task queue, the agent runner per step, each process tick); GET endpoints call `current` and write nothing.

### Adding a knob (WP18, WP19 and later)

1. Add a row to `core/controls.yaml` (with the failure it prevents, `where`, `status`, `set_by`; `consumer` names the code that reads it).
2. Add one typed property with the row's `accessor` name to `core/knobs.py`.
3. Add the row to docs/design/12-agent-controls.md (with its `set_by` cell); if an X experiment sets it, list the knob in that experiment's `sets`.
4. Read it through `ctx.controls.<accessor>` (pure code) or `kta.store.controls.ensure(...)`.<accessor>.

`test_registry_holds_every_row_of_12_with_set_by` and `test_every_knob_has_one_typed_accessor` fail until the three places agree. On the first run after the change, `ensure` writes a system version that adds the knob with its initial value and keeps every stored value.
