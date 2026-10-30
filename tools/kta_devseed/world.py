"""A scripted 55-minute afternoon ending at seed time, written through the real Store and Ops.

# kta:fixture synthetic loops, tasks, spans and evaluation records for frontend development
(D95, D168). Times are minutes from the start of the window; ``None`` means still running.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from kta.core.ids import UlidFactory
from kta.core.schemas import Actor
from kta.core.trace import SpanStart
from kta.store import SqliteOps
from kta_devseed.evaluation import seed_evaluation, seed_snapshots
from kta_devseed.knowledge import Seeder

WINDOW_MIN = 55
Steps = list[tuple[str, float, float | None, str]]

LOOPS: dict[str, tuple[str, str, str, list[Steps]]] = {
    # key: (machine, subject key, display name, rounds)
    "vadd": (
        "solve",
        "case:vadd",
        "vadd_f32",
        [
            [
                ("new", 11, 11.3, ""),
                ("bounded", 11.3, 11.7, ""),
                ("retrieving", 11.7, 12.2, "vadd_f32@1 legal"),
                ("solving", 12.2, 12.7, "shortlist 18"),
                ("building", 12.7, 15, "18 built, top 8"),
                ("checking", 15, 17, "8/8 passed"),
                ("measuring", 17, 22, ""),
                ("judging", 22, 22.5, "rule 3"),
                ("accepted", 22.5, None, ""),
            ]
        ],
    ),
    "softmax": (
        "solve",
        "case:softmax",
        "softmax_bf16",
        [
            [
                ("new", 3, 3.5, ""),
                ("bounded", 3.5, 4, ""),
                ("retrieving", 4, 5, "softmax_rowwise@1: precondition cols <= 512 fails"),
                ("waiting", 5, 32, "T1 extend family"),
            ],
            [
                ("retrieving", 32, 33, "softmax_rowwise@2 legal"),
                ("solving", 33, 34, "shortlist 24"),
                ("building", 34, 40, "24 built, top 8"),
                ("checking", 40, 44, "8/8 passed"),
                ("measuring", 44, None, "waited for the GPU lease, now measuring"),
            ],
        ],
    ),
    "gemm": (
        "solve",
        "case:gemm",
        "gemm_bf16_256",
        [
            [
                ("new", 15, 15.5, ""),
                ("bounded", 15.5, 16, ""),
                ("retrieving", 16, 16.7, "gemm_tiled@1 legal"),
                ("solving", 16.7, 18, "shortlist 64"),
                ("building", 18, 24, "61 built, 3 spill; top 8"),
                ("checking", 24, 28, "8/8 passed"),
                ("measuring", 28, 33, ""),
                ("judging", 33, 34, "rule 4: 38% of the gap unexplained"),
                ("waiting", 34, None, "T1 add smem double buffer"),
            ]
        ],
    ),
    "rmsnorm": (
        "solve",
        "case:rmsnorm",
        "rmsnorm_bf16",
        [
            [
                ("new", 43, 43.5, ""),
                ("bounded", 43.5, 44, ""),
                ("retrieving", 44, 45, "no legal Template"),
                ("waiting", 45, None, "T1 queued"),
            ]
        ],
    ),
    "triton": (
        "expand",
        "reference:triton_softmax_tutorial",
        "triton_softmax_tutorial",
        [
            [
                ("frozen", 9, 12, "built in container"),
                ("measured", 12, 20, "reference point 9.8 µs"),
                ("analyzing", 20, None, "T2 decomposing"),
            ]
        ],
    ),
    "cutlass": (
        "expand",
        "reference:cutlass_sgemm_example",
        "cutlass_sgemm_example",
        [[("analyzing", -40, -30, "T2: 4 of 7 steps matched"), ("gap", -30, None, "3 gaps")]],
    ),
    "cub": (
        "expand",
        "reference:cub_block_reduce",
        "cub_block_reduce",
        [
            [
                ("reproduced", -20, 37, "error 3.1% < 5%"),
                ("ablated", 37, 52, "4 ablations measured"),
                ("covered", 52, None, ""),
            ]
        ],
    ),
}


class SeedClock:
    def __init__(self, start: datetime) -> None:
        self.t = start

    def now(self) -> datetime:
        return self.t


class World:
    def __init__(self, s: Seeder, ops: SqliteOps, end: datetime) -> None:
        self.s, self.ops, self.end = s, ops, end
        self.t0 = end - timedelta(minutes=WINDOW_MIN)
        self.ulid = UlidFactory(s.clock)
        self.loop_ids = {k: self.ulid.new() for k in LOOPS}

    def at(self, minute: float) -> datetime:
        return self.t0 + timedelta(minutes=minute)

    def go(self, minute: float) -> None:
        self.s.clock.t = self.at(minute)

    def span(
        self,
        lane: str,
        kind: str,
        name: str,
        a: float,
        b: float | None,
        trace: str,
        attrs: dict[str, Any] | None = None,
        status: str = "ok",
        parent: str | None = None,
    ) -> str:
        sid = self.ulid.new()
        self.go(a)
        self.ops.start_span(
            SpanStart(sid, parent, trace, kind, lane, name, self.at(a), dict(attrs or {}))
        )
        if b is not None:
            self.go(b)
            self.ops.end_span(sid, self.at(b), status, {})  # type: ignore[arg-type]
        return sid

    def jobs(
        self,
        lane: str,
        kind: str,
        base: str,
        a: float,
        b: float,
        n: int,
        trace: str,
        pool: str,
        worker: str,
    ) -> None:
        step = (b - a) / n
        for i in range(n):
            jid = self.ulid.new()
            x = a + i * step
            q = min(step * 0.2, 0.3)
            common = {"job_id": jid, "job_kind": kind, "pool": pool, "worker": worker}
            self.span(lane, "job_queue", f"queue {base}#{i + 1}", x, x + q, trace, common)
            self.span(
                lane,
                kind,
                f"{base}#{i + 1}",
                x + q,
                x + step - 0.03,
                trace,
                common,
                status="fail" if base.startswith("nvcc gemm") and i == 2 else "ok",
            )


def seed_world(s: Seeder, ops: SqliteOps, end: datetime) -> None:
    w = World(s, ops, end)
    w.go(-60)
    _workers(w)
    _budgets(w)
    for key, (machine, _subject, name, rounds) in LOOPS.items():
        lane = "process:solve" if machine == "solve" else "process:learn"
        for rn, steps in enumerate(rounds, 1):
            for state, a, b, note in steps:
                if state in ("accepted", "covered"):
                    continue
                w.span(
                    lane,
                    "loop_step",
                    f"{name} · {state}",
                    max(a, -60),
                    b,
                    w.loop_ids[key],
                    {"state": state, "round": rn, "note": note},
                )
    tasks = _tasks(w)
    _jobs(w, tasks)
    _evaluation(w, tasks)
    _snapshots(w)
    for key, (machine, subject, name, rounds) in LOOPS.items():
        last = rounds[-1][-1]
        w.go(last[1])
        data: dict[str, Any] = {
            "name": name,
            "started_at": _iso(w.at(rounds[0][0][1])),
            "state_since": _iso(w.at(last[1])),
        }
        if machine == "solve":
            data["round"] = len(rounds)
        waiting = {
            "gemm": "t1-gemm",
            "triton": "t2-triton",
            "cutlass": "t3-swizzle",
            "rmsnorm": "t1-rms",
        }.get(key)
        if waiting:
            data["waiting_on"] = tasks[waiting]
        if key == "cutlass":
            data["escalated"] = False
        ops.put_loop(w.loop_ids[key], machine, s.ids[subject], last[0], data)
    w.go(WINDOW_MIN)


def _iso(t: datetime) -> str:
    return t.isoformat().replace("+00:00", "Z")


def _workers(w: World) -> None:
    s = w.s
    for host, arch in (("spark", "aarch64"), ("slai", "x86_64")):
        s.put(
            f"env:{host}",
            "EnvManifest",
            {
                "role": "worker",
                "host": host,
                "os": "Linux",
                "arch": arch,
                "python": "3.12.13",
                "uv": "0.11.2",
                "uv_lock_sha256": "fixture",
                "commit": "fixture",
                "dirty": False,
            },
        )
    s.put(
        "worker:spark",
        "Worker",
        {
            "name": "spark-gpu0",
            "host": "spark",
            "transport": "ssh",
            "devices": ["0"],
            "commit": "fixture",
            "labels": {
                "gpu": "sm_121",
                "cuda": "13.0",
                "pools": "gpu-exclusive:1,gpu-shared:2,cpu:18",
            },
        },
    )
    s.put(
        "worker:slai",
        "Worker",
        {
            "name": "slai-cpu",
            "host": "slai",
            "transport": "ssh",
            "commit": "fixture",
            "labels": {"arch": "x86_64", "pools": "cpu:30"},
        },
    )
    w.go(WINDOW_MIN - 0.1)
    w.ops.heartbeat("spark-gpu0", "spark", True)
    w.ops.heartbeat("slai-cpu", "slai", True)
    w.go(-60)


def _budgets(w: World) -> None:
    o = w.ops
    o.set_budget("global", {"agent_min": 480, "gpu_min": 120})
    for scope, agent, gpu in (
        ("process:solve", 240, 60),
        ("process:learn", 144, 36),
        ("process:calibrate", 96, 24),
    ):
        o.set_budget(scope, {"agent_min": agent, "gpu_min": gpu})
    for scope, used, reserved in (
        ("global", {"agent_min": 176, "gpu_min": 41}, {"agent_min": 105, "gpu_min": 10}),
        ("process:solve", {"agent_min": 47, "gpu_min": 27}, {"agent_min": 45, "gpu_min": 10}),
        ("process:learn", {"agent_min": 109, "gpu_min": 14}, {"agent_min": 30}),
    ):
        if not o.reserve(scope, {k: used[k] + reserved.get(k, 0) for k in used}):
            raise ValueError(f"fixture budget for {scope} exceeds its limit")
        o.settle(scope, used, used)


TASKS = [
    # key, type, loop, goal, created, leased, ended, final state, slot, priority, budget min
    (
        "t3-swizzle",
        "T3",
        "cutlass",
        "Extract a Template for the swizzled smem layout used with ldmatrix",
        -29,
        -28,
        1,
        "failed",
        "agent:2",
        2,
        20,
    ),
    (
        "t3-epilogue",
        "T3",
        "cutlass",
        "Extract a Template for the predicated epilogue on ragged tiles",
        1,
        None,
        None,
        None,
        None,
        3,
        20,
    ),
    (
        "t1-softmax",
        "T1",
        "softmax",
        "Extend softmax_rowwise so cols = 1024 is legal (online softmax)",
        5,
        6,
        32,
        "done",
        "agent:1",
        1,
        45,
    ),
    (
        "t2-triton",
        "T2",
        "triton",
        "Decompose triton_softmax_tutorial into a composition chain",
        20,
        21,
        None,
        None,
        "agent:2",
        2,
        60,
    ),
    (
        "t1-rms",
        "T1",
        "rmsnorm",
        "Build a family for rmsnorm_bf16 from row_per_cta and warp_reduce",
        45,
        None,
        None,
        None,
        None,
        2,
        45,
    ),
    (
        "t1-gemm",
        "T1",
        "gemm",
        "Extend gemm_tiled with smem_double_buffer to hide global-load latency",
        34,
        35,
        None,
        None,
        "agent:1",
        1,
        45,
    ),
]

THINK = [
    ("think", 1.2),
    ("kta show", 0.1),
    ("kta query", 0.15),
    ("think", 2.5),
    ("kta build", 1.4),
    ("think", 1.6),
    ("kta check", 2.2),
    ("think", 2.7),
    ("kta build", 1.2),
    ("kta check", 2.0),
    ("think", 2.0),
    ("kta validate-family", 0.4),
    ("kta submit", 0.1),
]


def _tasks(w: World) -> dict[str, str]:
    s, o = w.s, w.ops
    ids: dict[str, str] = {}
    # Enqueue in time order: Ops.lease takes the highest-priority ready task of a type.
    for key, ttype, loop, goal, created, leased, ended, final, slot, prio, budget in sorted(
        TASKS, key=lambda t: t[4]
    ):
        w.go(created)
        card = s.put(
            f"card:{key}",
            "TaskCard",
            {
                "type": ttype,
                "goal": goal,
                "inputs": [],
                "profile": "fixture-profile",
                "work_unit": w.loop_ids[loop],
                "budget": {"wall_s": budget * 60, "gpu_s": 300, "steps": 80},
                "done_checks": ["delivery_note_complete"],
            },
        )
        tid = o.enqueue(card, prio, w.loop_ids[loop])
        ids[key] = tid
        if leased is None or slot is None:
            continue
        w.go(leased)
        if key == "t3-swizzle":
            for i in range(3):  # three attempts: two retries after rejections, then over the limit
                o.lease(slot, [ttype], -1 if i < 2 else 600)
                w.go(leased + 8 * (i + 1))
                s.put(
                    f"reject:{key}:{i}",
                    "Check",
                    {
                        "passed": False,
                        "failures": [
                            {
                                "check": "validate_template",
                                "error": [
                                    "variable 'swz' has no domain",
                                    "precondition references undeclared name hw.ldmatrix_x4",
                                    "precondition references undeclared name hw.ldmatrix_x4",
                                ][i],
                            }
                        ],
                    },
                    refs={"env": s.ids["env:spark"], "worker": s.ids["worker:spark"]},
                    actor=Actor(type="checker", name="validate_template@1", task=tid),
                )
        else:
            o.lease(slot, [ttype], 3600)
            sess = w.span(
                slot,
                "agent_session",
                f"{ttype} {key.split('-', 1)[1]}",
                leased,
                ended - 0.5 if ended else None,
                w.loop_ids[loop],
                {"task_id": tid, "session": w.ulid.new()},
            )
            _session_children(
                w, slot, sess, leased, ended - 0.5 if ended else None, w.loop_ids[loop], tid
            )
        if ended is not None and final is not None:
            w.go(ended)
            o.finish(tid, final)
    return ids


def _session_children(
    w: World, lane: str, parent: str, a: float, b: float | None, trace: str, tid: str
) -> None:
    t = a
    stop = b if b is not None else WINDOW_MIN
    while t < stop:
        for name, dur in THINK:
            if t >= stop:
                return
            e = min(t + dur, stop)
            open_ = b is None and e >= stop
            kind = "agent_model" if name == "think" else "tool_call"
            w.span(
                lane,
                kind,
                "model" if name == "think" else name,
                t,
                None if open_ else e,
                trace,
                {"task_id": tid},
                parent=parent,
            )
            if name in ("kta build", "kta check"):
                lane2 = "worker:slai:cpu" if name == "kta build" else "worker:spark:gpu-shared"
                w.span(
                    lane2,
                    "compile" if name == "kta build" else "check",
                    f"{name[4:]} ({tid[-6:]})",
                    t + 0.05,
                    None if open_ else e - 0.02,
                    trace,
                    {
                        "task_id": tid,
                        "job_id": w.ulid.new(),
                        "pool": "cpu" if name == "kta build" else "gpu-shared",
                        "worker": "slai-cpu" if name == "kta build" else "spark-gpu0",
                    },
                )
            t = e


def _jobs(w: World, tasks: dict[str, str]) -> None:
    L = w.loop_ids
    w.jobs(
        "worker:slai:cpu",
        "compile",
        "docker build triton",
        9.2,
        11.8,
        1,
        L["triton"],
        "cpu",
        "slai-cpu",
    )
    w.span(
        "worker:spark:gpu-shared",
        "check",
        "check triton_softmax",
        12.2,
        14.8,
        L["triton"],
        {"pool": "gpu-shared", "worker": "spark-gpu0"},
    )
    w.span(
        "gpu:spark-gpu0:0",
        "measure",
        "measure triton_softmax",
        15,
        17,
        L["triton"],
        {"protocol": "measure-v1", "worker": "spark-gpu0"},
    )
    w.jobs("worker:spark:cpu", "compile", "nvcc vadd", 12.7, 15, 4, L["vadd"], "cpu", "spark-gpu0")
    w.span(
        "worker:spark:gpu-shared",
        "check",
        "check vadd top8",
        15.1,
        16.9,
        L["vadd"],
        {"pool": "gpu-shared", "worker": "spark-gpu0"},
    )
    w.span(
        "gpu:spark-gpu0:0",
        "measure",
        "measure vadd top8",
        17.2,
        22,
        L["vadd"],
        {"protocol": "measure-v1", "worker": "spark-gpu0"},
    )
    w.jobs("worker:slai:cpu", "compile", "nvcc gemm", 18, 24, 6, L["gemm"], "cpu", "slai-cpu")
    w.span(
        "worker:spark:gpu-shared",
        "check",
        "check gemm top8",
        24.1,
        27.9,
        L["gemm"],
        {"pool": "gpu-shared", "worker": "spark-gpu0"},
    )
    w.span(
        "gpu:spark-gpu0:0",
        "measure",
        "measure gemm top8",
        28.2,
        33,
        L["gemm"],
        {"protocol": "measure-v1", "worker": "spark-gpu0"},
    )
    w.jobs(
        "worker:spark:cpu",
        "compile",
        "nvcc softmax@2",
        34,
        40,
        5,
        L["softmax"],
        "cpu",
        "spark-gpu0",
    )
    w.span(
        "worker:spark:gpu-shared",
        "check",
        "check softmax top8",
        40.1,
        43.9,
        L["softmax"],
        {"pool": "gpu-shared", "worker": "spark-gpu0"},
    )
    for i, what in enumerate(["warp_reduce", "smem_pad", "unroll", "vector_load"]):
        a = 37 + i * 3.75
        w.span(
            "gpu:spark-gpu0:0",
            "measure",
            f"ablation cub #{i + 1}",
            a,
            a + 3.6,
            L["cub"],
            {"ablate": what, "worker": "spark-gpu0"},
        )
    job = w.ulid.new()
    common = {"job_id": job, "job_kind": "measure", "pool": "gpu-exclusive", "worker": "spark-gpu0"}
    w.span("gpu:spark-gpu0:0", "job_queue", "queue measure softmax", 44, 52, L["softmax"], common)
    w.span(
        "gpu:spark-gpu0:0",
        "lock_hold",
        "gpu lease",
        52,
        None,
        L["softmax"],
        {"lock": "gpu:spark-gpu0:0", "job_id": job},
    )
    w.span("gpu:spark-gpu0:0", "measure", "measure softmax top8", 52, None, L["softmax"], common)
    w.go(52)
    w.ops.acquire("gpu:spark-gpu0:0", job, 600)
    w.span(
        "worker:spark:gpu-shared",
        "job_queue",
        "queue check gemm@2",
        52.2,
        None,
        L["gemm"],
        {
            "job_id": w.ulid.new(),
            "job_kind": "check",
            "pool": "gpu-shared",
            "worker": "spark-gpu0",
            "task_id": tasks["t1-gemm"],
            "waits_for": "exclusive lease released",
        },
    )
    w.span(
        "human",
        "human_wait",
        "inbox: T3 swizzle escalated",
        1,
        None,
        w.ulid.new(),
        {"inbox": "escalated task"},
    )


def _evaluation(w: World, tasks: dict[str, str]) -> None:
    seed_evaluation(w.s, w.go)


def _snapshots(w: World) -> None:
    seed_snapshots(w.s, w.ops, w.go)
