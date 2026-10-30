# targets、execution、backends

设计依据：docs/design/11 §2–§6、40 §2–§4、50 §6。

## targets

### 文件

```
targets/base.py          Target 协议
targets/gb10/            第一个实现：probe.py、limits.py、units.py、features.yaml
targets/microbench/      microbenchmark 源码与运行脚本（由 T5 或 WP 编写）
```

### 接口

```python
class TargetPlugin(Protocol):
    name: str
    def probe(self, ex: "Executor") -> Target            # 远端读取 GPU、驱动、CUDA、ptxas、镜像、后端版本
    def units(self) -> list[str]                         # 硬件单元名
    def limits(self) -> dict[str, int | float]           # 表达式中 hw.* 的取值：smem_per_cta、regs_per_thread、max_threads_per_cta 等
    def features(self) -> list[HwFeature]                # 初始清单来自 features.yaml，support 由探测更新
    def compile_flags(self, backend: str) -> list[str]
```

### 行为

- `features.yaml` 是开放的清单。初始内容按 PTX ISA 文档整理指令族（各类 MMA、cp.async、TMA、mbarrier、各级 load/store、原子、warp 级原语等），每项 `support` 初值为 `unknown`。探测脚本对每项编译并运行一个最小 kernel，成功则 `yes`，编译失败则 `no`，记录原始输出。
- `limits` 中的数值，先取 CUDA 设备属性查询的结果，并写成 HwFact（method="documented"）；microbenchmark 实测后另写 HwFact（method="measured"）。

## execution

### 文件

```
execution/base.py        Executor 协议、Job、JobResult
execution/registry.py    worker 注册、心跳、能力标签匹配
execution/transport/     local.py、ssh.py、pull.py（pull 第一阶段抛 NotSupported，协议写在 pull.md）
execution/worker.py      worker 侧：执行作业、维护本地产物缓存、执行 GPU 独占租约；`kta worker` 守护进程入口
execution/blobs.py       产物上传、下载、校验
execution/sandbox.py     容器内执行（外部代码的构建与运行）
scripts/worker-setup     在一台机器上按当前 commit 部署 worker 环境、探测能力、注册
```

### 接口

```python
class Job:
    kind: Literal["compile", "run", "measure", "profile", "check", "microbench", "shell"]
    requires: dict[str, str]      # 能力标签，例如 {"gpu": "sm_121", "pool": "gpu-exclusive"}
    argv: list[str]; env: dict[str, str]
    inputs: dict[str, str]        # 名 → 产物 sha256；在 worker 上放进作业目录
    timeout_s: int; mem_limit_mb: int | None
    container: str | None         # 镜像 digest；None 表示 worker 的宿主环境
class JobResult:
    ok: bool; exit_code: int; wall_s: float; queue_s: float
    worker: str; device: str | None; env_manifest: str    # 记录 ID
    stdout_blob: str; stderr_blob: str; outputs: dict[str, str]
    error_kind: Literal["none", "timeout", "oom", "gpu_hang", "infra", "nonzero"]

class Executor(Protocol):
    def submit(self, job: Job) -> str
    def wait(self, job_id: str) -> JobResult
    def run(self, job: Job) -> JobResult
    def quiet_lease(self, worker: str, device: str, holder: str, max_s: int = 600) -> ContextManager[None]
def register_worker(ctx, name: str, host: str, transport: str) -> Worker      # 探测能力标签，写 Worker 与 EnvManifest
```

### 行为

- 控制面与 worker 可以在不同机器、不同集群，不共享文件系统（conventions §6）。作业的输入输出只以产物（按 sha256）传递，传输后校验。
- 作业按 `requires` 匹配 worker；没有满足的 worker 时，作业留在队列中，并在待处理中提示。
- worker 的代码 commit 必须与控制面一致，否则不派发作业。
- 执行池并发上限见 docs/design/11 §4.1，由配置为每个 worker 设定。
- `gpu-exclusive` 作业在对应 worker 上取得该 GPU 设备的独占租约：停止派发新的 gpu-shared 作业，等待正在运行的结束，暂停该 worker 上的编译作业。最长 10 分钟，超时强制释放并写 ops_events。
- 超时终止整个进程组。GPU 无响应判为 `gpu_hang`，标记 worker 不健康。
- stdout、stderr 一律存为产物；业务代码不得把原始大输出直接塞进记录或提示。
- 每个作业有 `job_queue` 与 `job_run` 两个 span，泳道为 `worker:<名>:<执行池>`。

## backends

### 文件

```
backends/base.py        Backend 协议、GeneratedKernel、BuiltModule
backends/tilelang/      TileLang 后端
backends/cuda/          CUDA C++ 后端（nvcc；可用 CUTLASS 头文件）
backends/harness/       统一的 C ABI 加载器与计时入口（被 measure 使用）
```

### 接口

```python
class GeneratedKernel:
    backend: str
    files: dict[str, str]          # 文件名 → 源码文本
    entry: str                     # 符号名
    launch: LaunchSpec             # grid、block、动态 smem、参数顺序与类型
    markers: list[str]             # 源码中出现的 kta:step 标记，用于一致性检查
class BuiltModule:
    sha256: str                    # .so 或 cubin 的 blob
    kind: Literal["so", "cubin"]
    feedback: dict                 # regs、spills、smem、ptxas 摘要
    sass_blob: str | None          # cuobjdump -sass 的输出
class Backend(Protocol):
    name: str
    def supports(self, target: Target) -> bool
    def build(self, gk: GeneratedKernel, target: Target, ex: Executor) -> tuple[Build, BuiltModule]
```

### 行为

- 编译按输入 hash 缓存：同样的源码、同样的 Target、同样的编译参数，直接返回已有的 Build。
- 所有后端的产物都通过 `harness` 用同一种方式加载和启动，保证计时口径一致。TileLang 后端导出 CUDA 源码或 cubin 后，同样走 harness。

## 验收（三个模块合计）

- 真实运行：本机控制面经 `ssh` 传输把 spark 注册为 GPU worker（`scripts/worker-setup spark`），完成 `probe`，写出 Worker、EnvManifest、Target 记录和 features 的支持情况；用两个后端各编译并运行一个向量加法 kernel，输出正确；`gpu-exclusive` 作业期间，并发提交的 gpu-shared 作业被推迟（日志可见）；故意构造一个死循环 kernel，验证超时后 GPU 恢复可用。
- 重新部署：在第二台机器（slai）上用 `scripts/bootstrap` 建立控制面，或用 `scripts/worker-setup slai` 注册 CPU worker，跑通单元测试和一个编译作业；两台机器的 EnvManifest 中锁文件 hash 一致。
- 交付说明写明：各机器 `KTA_DATA` 的取值、CUDA 与驱动版本、TileLang 版本、哪些 features 为 yes/no、slai 能否为 sm_121 交叉编译。
