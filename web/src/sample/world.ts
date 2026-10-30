// kta:fixture sample data before the store exists (D95); deleted in WP9a phase B.
// One scripted afternoon (13:35-14:30 UTC) from which loops, tasks, spans and case timelines
// are all derived, so every page tells the same story.

import type { LoopInstance, RoundStep, Span, TaskInstance, TaskType, TraceInfo } from "../data/types";
import { AS_OF, at, ulid } from "./ids";
import { agent, put, tool } from "./records";

// ---------------------------------------------------------------- loop scripts

type Step = [state: string, start: string, end: string | null, note?: string];

interface LoopScript {
  key: string;
  machine: "solve" | "expand";
  subject_name: string;
  rounds: Step[][];
}

export const LOOP_SCRIPTS: LoopScript[] = [
  { key: "vadd", machine: "solve", subject_name: "vadd_f32", rounds: [[
    ["new", "13:46:00", "13:46:20"], ["bounded", "13:46:20", "13:46:40"], ["retrieving", "13:46:40", "13:47:10", "vadd_f32@1 legal"],
    ["solving", "13:47:10", "13:47:40", "shortlist 18"], ["building", "13:47:40", "13:50:00", "18 built, top 8"],
    ["checking", "13:50:00", "13:52:00", "8/8 passed"], ["measuring", "13:52:00", "13:57:00"],
    ["judging", "13:57:00", "13:57:30", "rule 3: within explained gap"], ["accepted", "13:57:30", null],
  ]] },
  { key: "softmax", machine: "solve", subject_name: "softmax_bf16", rounds: [
    [["new", "13:38:00", "13:38:30"], ["bounded", "13:38:30", "13:39:00"],
     ["retrieving", "13:39:00", "13:40:00", "softmax_rowwise@1: precondition cols <= 512 fails"], ["waiting", "13:40:00", "14:07:00", "T1 extend family"]],
    [["retrieving", "14:07:00", "14:08:00", "softmax_rowwise@2 legal"], ["solving", "14:08:00", "14:09:00", "shortlist 24"],
     ["building", "14:09:00", "14:15:00", "24 built, top 8"], ["checking", "14:15:00", "14:19:00", "8/8 passed"],
     ["measuring", "14:19:00", null, "waiting for GPU exclusive lease, then measuring"]],
  ] },
  { key: "gemm", machine: "solve", subject_name: "gemm_bf16_256", rounds: [[
    ["new", "13:50:00", "13:50:30"], ["bounded", "13:50:30", "13:51:00"], ["retrieving", "13:51:00", "13:51:40", "gemm_tiled@1 legal"],
    ["solving", "13:51:40", "13:53:00", "shortlist 64"], ["building", "13:53:00", "13:59:00", "61 built, 3 spill; top 8"],
    ["checking", "13:59:00", "14:03:00", "8/8 passed"], ["measuring", "14:03:00", "14:08:00"],
    ["judging", "14:08:00", "14:09:00", "rule 4: 38% of gap unexplained"], ["waiting", "14:09:00", null, "T1 add smem double buffer"],
  ]] },
  { key: "rmsnorm", machine: "solve", subject_name: "rmsnorm_bf16", rounds: [[
    ["new", "14:18:00", "14:18:30"], ["bounded", "14:18:30", "14:19:00"], ["retrieving", "14:19:00", "14:20:00", "rmsnorm_fused@1 legal"],
    ["solving", "14:20:00", "14:22:00", "shortlist 12"], ["building", "14:22:00", null, "5 of 12 built"],
  ]] },
  { key: "triton_softmax", machine: "expand", subject_name: "triton_softmax_tutorial", rounds: [[
    ["frozen", "13:44:00", "13:47:00", "commit a1f3c2e, built in container"], ["measured", "13:47:00", "13:55:00", "reference point 9.8 µs"],
    ["analyzing", "13:55:00", null, "T2 decomposing"],
  ]] },
  { key: "cutlass_sgemm", machine: "expand", subject_name: "cutlass_sgemm_example", rounds: [[
    ["frozen", "12:05:00", "12:09:00"], ["measured", "12:09:00", "12:20:00"], ["analyzing", "12:20:00", "13:05:00", "T2: 4 of 7 steps matched"],
    ["gap", "13:05:00", null, "3 gaps; 1 filled, 1 escalated, 1 queued"],
  ]] },
  { key: "cub_reduce", machine: "expand", subject_name: "cub_block_reduce", rounds: [[
    ["frozen", "11:10:00", "11:14:00"], ["measured", "11:14:00", "11:25:00"], ["analyzing", "11:25:00", "11:58:00", "T2: 5 of 5 matched"],
    ["reconstructed", "11:58:00", "12:10:00"], ["reproduced", "12:10:00", "14:12:00", "error 3.1% < ε 5%"],
    ["ablated", "14:12:00", "14:27:00", "4 ablations measured"], ["covered", "14:27:00", null],
  ]] },
];

export const loopId = (key: string) => ulid(`loop:${key}`);
export const subjectId = (machine: string, name: string) =>
  put(machine === "solve" ? "case" : "reference", name, { name });

// ---------------------------------------------------------------- tasks

interface TaskScript {
  key: string;
  type: TaskType;
  loop: string;
  goal: string;
  created: string;
  leased?: string;
  ended?: string;
  state: string;
  slot?: string;
  retries?: number;
  budget_min: number;
  priority: number;
  rejection?: string;
}

export const TASK_SCRIPTS: TaskScript[] = [
  { key: "t1-softmax", type: "T1", loop: "softmax", goal: "Extend softmax_rowwise so cols = 1024 is legal (online softmax)",
    created: "13:40:00", leased: "13:41:00", ended: "14:07:00", state: "accepted", slot: "agent:1", budget_min: 45, priority: 1 },
  { key: "t1-gemm", type: "T1", loop: "gemm", goal: "Extend gemm_tiled with smem_double_buffer to hide global-load latency",
    created: "14:09:00", leased: "14:10:00", state: "running", slot: "agent:1", budget_min: 45, priority: 1 },
  { key: "t2-triton", type: "T2", loop: "triton_softmax", goal: "Decompose triton_softmax_tutorial into a composition chain",
    created: "13:55:00", leased: "13:56:00", state: "running", slot: "agent:2", budget_min: 60, priority: 2 },
  { key: "t3-cutlass-swizzle", type: "T3", loop: "cutlass_sgemm", goal: "Extract a Template for the swizzled smem layout used with ldmatrix",
    created: "13:06:00", leased: "13:07:00", ended: "13:36:00", state: "escalated", retries: 2, budget_min: 20, priority: 2,
    rejection: "validate_template: precondition references undeclared name hw.ldmatrix_x4" },
  { key: "t3-cutlass-epilogue", type: "T3", loop: "cutlass_sgemm", goal: "Extract a Template for the predicated epilogue on ragged tiles",
    created: "13:36:00", state: "queued", budget_min: 20, priority: 3 },
];

export const taskId = (key: string) => ulid(`task:${key}`);

export function taskCard(s: TaskScript): string {
  return put("task_card", s.key, {
    type: s.type, goal: s.goal, inputs: [subjectOf(s.loop)], done_checks: DONE_CHECKS[s.type] ?? [],
    budget: { wall_s: s.budget_min * 60, gpu_s: 300, steps: 80 }, profile: "codex-default@3", work_unit: loopId(s.loop),
  }, { actor: { type: "system", name: "orchestrator" }, created_at: at(s.created) });
}

const DONE_CHECKS: Record<string, string[]> = {
  T1: ["validate_family", "verify_space", "delivery_note_complete"],
  T2: ["reconstruction_fields", "matched_items_reference_templates", "gaps_match_unmatched", "delivery_note_complete"],
  T3: ["validate_template", "delivery_note_complete"],
};

function subjectOf(loopKey: string): string {
  const l = LOOP_SCRIPTS.find((x) => x.key === loopKey)!;
  return subjectId(l.machine, l.subject_name);
}

const sec = (a: string, b: string) => (Date.parse(b) - Date.parse(a)) / 1000;

export function buildTasks(): TaskInstance[] {
  return TASK_SCRIPTS.map((s) => {
    const leased = s.leased ? at(s.leased) : undefined;
    const end = s.ended ? at(s.ended) : AS_OF;
    return {
      id: taskId(s.key), card: taskCard(s), type: s.type, goal: s.goal, state: s.state, slot: s.state === "running" ? s.slot : undefined,
      loop_id: loopId(s.loop), created_at: at(s.created), state_since: s.ended ? at(s.ended) : leased ?? at(s.created),
      retries: s.retries ?? 0, max_retries: 2, budget_wall_s: s.budget_min * 60, used_wall_s: leased ? sec(leased, end) : 0,
      queue_s: sec(at(s.created), leased ?? AS_OF), priority: s.priority, last_rejection: s.rejection,
    };
  });
}

export function buildLoops(): LoopInstance[] {
  return LOOP_SCRIPTS.map((l) => {
    const last = l.rounds[l.rounds.length - 1];
    const cur = last[last.length - 1];
    const waitingTask = TASK_SCRIPTS.find((t) => t.loop === l.key && ["running", "queued", "escalated"].includes(t.state));
    return {
      id: loopId(l.key), machine: l.machine, subject: subjectOf(l.key), subject_name: l.subject_name, state: cur[0],
      round: l.machine === "solve" ? l.rounds.length : undefined, max_rounds: l.machine === "solve" ? 5 : undefined,
      started_at: at(l.rounds[0][0][1]), state_since: at(cur[1]),
      waiting_on: ["waiting", "analyzing", "gap"].includes(cur[0]) ? (waitingTask ? taskId(waitingTask.key) : undefined) : undefined,
      escalated: l.key === "cutlass_sgemm", trace_id: loopId(l.key),
    };
  });
}

export function roundSteps(key: string): RoundStep[][] {
  const l = LOOP_SCRIPTS.find((x) => x.key === key)!;
  return l.rounds.map((r) => r.map(([state, s, e, note]) => ({ state, start: at(s), end: e ? at(e) : null, note })));
}

export const TRACES: TraceInfo[] = [
  ...LOOP_SCRIPTS.map((l) => ({ id: loopId(l.key), label: `${l.machine === "solve" ? "Case" : "Reference"} ${l.subject_name}`, kind: l.machine })),
  { id: ulid("trace:human"), label: "Human", kind: "human" as const },
];

// ---------------------------------------------------------------- spans

const SPANS: Span[] = [];
let seq = 0;

function sp(lane: string, kind: string, name: string, start: string, end: string | null, trace: string,
  attrs: Span["attrs"] = {}, status?: Span["status"], parent: string | null = null): string {
  const span_id = ulid(`span:${seq++}`);
  SPANS.push({ span_id, parent_id: parent, trace_id: trace, kind, lane, name, start_ts: at(start), end_ts: end ? at(end) : null,
    status: status ?? (end ? "ok" : "running"), attrs });
  return span_id;
}

function hms(t: number): string {
  const d = new Date(t);
  return `${d.getUTCHours()}:${String(d.getUTCMinutes()).padStart(2, "0")}:${String(d.getUTCSeconds()).padStart(2, "0")}`;
}
const ms = (hhmmss: string) => Date.parse(at(hhmmss));

/** Split [start, end) into `n` back-to-back jobs on a lane, each preceded by a short queue span. */
function jobs(lane: string, kind: string, base: string, start: string, end: string, n: number, trace: string, extra: Span["attrs"] = {}) {
  const t0 = ms(start);
  const step = (ms(end) - t0) / n;
  for (let i = 0; i < n; i++) {
    const a = t0 + i * step;
    const q = Math.min(step * 0.2, 20_000);
    sp(lane, "job_queue", `queue ${base}#${i + 1}`, hms(a), hms(a + q), trace, { job_id: ulid(`job:${base}:${i}`) });
    sp(lane, kind, `${base}#${i + 1}`, hms(a + q), hms(a + step - 2000), trace, { job_id: ulid(`job:${base}:${i}`), ...extra },
      kind === "compile" && i === 2 && base.startsWith("nvcc gemm") ? "fail" : undefined);
  }
}

// Loop steps on the owning process lane.
for (const l of LOOP_SCRIPTS) {
  const lane = l.machine === "solve" ? "process:solve" : "process:learn";
  l.rounds.forEach((r, ri) => r.forEach(([state, s, e, note]) => {
    if (ms(s) < ms("13:35:00") || state === "accepted" || state === "covered") return;
    sp(lane, "loop_step", `${l.subject_name} · ${state}`, s, e, loopId(l.key), { state, round: ri + 1, ...(note ? { note } : {}) });
  }));
}

/** An agent session with alternating think / tool segments; tool calls that run jobs also land on worker lanes. */
function session(slot: string, taskKey: string, start: string, end: string | null, trace: string, script: [string, number][]) {
  const tid = taskId(taskKey);
  const sid = sp(`agent:${slot}`, "agent_session", `${TASK_SCRIPTS.find((t) => t.key === taskKey)!.type} ${taskKey.split("-").slice(1).join("-")}`,
    start, end, trace, { task_id: tid, session: ulid(`sess:${taskKey}`) });
  let t = ms(start);
  const stop = end ? ms(end) : Date.parse(AS_OF);
  for (const [name, secs] of script) {
    if (t >= stop) break;
    const e = Math.min(t + secs * 1000, stop);
    const open = !end && e === stop;
    const kind = name === "think" ? "agent_model" : "tool_call";
    sp(`agent:${slot}`, kind, name === "think" ? "model" : name, hms(t), open ? null : hms(e), trace, { task_id: tid }, undefined, sid);
    if (name.startsWith("kta build")) sp("worker:slai:cpu", "compile", `nvcc ${name.slice(10)}`, hms(t + 3000), open ? null : hms(e - 1000), trace, { task_id: tid });
    if (name.startsWith("kta check")) sp("worker:spark:gpu-shared", "check", name.slice(4), hms(t + 4000), open ? null : hms(e - 1000), trace, { task_id: tid });
    t = e;
  }
}

const T1_SOFTMAX: [string, number][] = [
  ["think", 70], ["kta show softmax_rowwise@1", 6], ["kta query template --refines reduce", 8], ["think", 140], ["kta why online_softmax@1", 5],
  ["think", 210], ["kta instantiate softmax@2", 12], ["kta build softmax@2", 85], ["think", 95], ["kta check softmax@2", 130], ["think", 160],
  ["kta build softmax@2", 70], ["kta check softmax@2", 120], ["think", 120], ["kta validate-family softmax@2", 25], ["think", 60],
  ["kta note", 4], ["kta submit", 6],
];
const T1_GEMM: [string, number][] = [
  ["think", 80], ["kta show gemm_tiled@1", 6], ["kta show smem_double_buffer@1", 5], ["think", 190], ["kta src cutlass_sgemm:mainloop", 10],
  ["think", 160], ["kta instantiate gemm@2", 14], ["kta build gemm@2", 95], ["think", 120], ["kta check gemm@2", 140], ["think", 170],
  ["kta build gemm@2", 80], ["kta check gemm@2", 300],
];
const T2_TRITON: [string, number][] = [
  ["think", 90], ["kta src triton_softmax:kernel", 12], ["think", 220], ["kta query pattern reduce", 7], ["kta show online_softmax@1", 5],
  ["think", 300], ["kta show row_per_cta@1", 4], ["think", 260], ["kta sass triton_softmax", 40], ["think", 380], ["kta query template --level 3", 9],
  ["think", 300], ["kta note", 5], ["think", 900],
];

session("1", "t1-softmax", "13:41:00", "14:04:00", loopId("softmax"), T1_SOFTMAX);
session("1", "t1-gemm", "14:10:00", null, loopId("gemm"), T1_GEMM);
session("2", "t2-triton", "13:56:00", null, loopId("triton_softmax"), T2_TRITON);

// Done-checks of the softmax T1 (validate_family + verify_space) run as jobs.
jobs("worker:spark:cpu", "compile", "nvcc softmax@2 space", "14:04:10", "14:06:00", 3, loopId("softmax"));
sp("worker:spark:gpu-shared", "check", "verify_space softmax@2", "14:06:00", "14:06:55", loopId("softmax"), {}, "ok");

// Executor jobs of the solve and expand loops.
jobs("worker:slai:cpu", "compile", "docker build triton_softmax", "13:44:10", "13:46:50", 1, loopId("triton_softmax"));
sp("worker:spark:gpu-shared", "check", "check triton_softmax", "13:47:10", "13:49:50", loopId("triton_softmax"));
sp("gpu:spark:0", "lock_wait", "lease wait", "13:49:50", "13:50:00", loopId("triton_softmax"));
sp("gpu:spark:0", "measure", "measure triton_softmax", "13:50:00", "13:52:00", loopId("triton_softmax"), { protocol: "measure-v1" });

jobs("worker:spark:cpu", "compile", "nvcc vadd", "13:47:40", "13:50:00", 4, loopId("vadd"));
sp("worker:spark:gpu-shared", "check", "check vadd top8", "13:50:05", "13:51:55", loopId("vadd"));
sp("gpu:spark:0", "measure", "measure vadd top8", "13:52:10", "13:57:00", loopId("vadd"), { protocol: "measure-v1", candidates: 8 });

jobs("worker:slai:cpu", "compile", "nvcc gemm", "13:53:00", "13:59:00", 6, loopId("gemm"));
sp("worker:spark:gpu-shared", "check", "check gemm top8", "13:59:05", "14:02:55", loopId("gemm"));
sp("gpu:spark:0", "measure", "measure gemm top8", "14:03:10", "14:08:00", loopId("gemm"), { protocol: "measure-v1", candidates: 8 });

jobs("worker:spark:cpu", "compile", "nvcc softmax@2", "14:09:00", "14:15:00", 5, loopId("softmax"));
sp("worker:spark:gpu-shared", "check", "check softmax top8", "14:15:05", "14:18:55", loopId("softmax"));
sp("gpu:spark:0", "job_queue", "queue measure softmax", "14:19:00", "14:27:00", loopId("softmax"), { waits_for: "gpu-exclusive lease" });
sp("gpu:spark:0", "measure", "measure softmax top8", "14:27:00", null, loopId("softmax"), { protocol: "measure-v1", candidates: 8, job_id: ulid("job:measure-softmax") });

for (let i = 0; i < 4; i++) {
  const a = ms("14:12:00") + i * 225_000;
  sp("gpu:spark:0", "measure", `ablation cub_reduce #${i + 1}`, hms(a), hms(a + 215_000), loopId("cub_reduce"), { ablate: ["warp_reduce", "smem_pad", "unroll", "vector_load"][i] });
}

jobs("worker:spark:cpu", "compile", "nvcc rmsnorm", "14:22:00", "14:28:30", 5, loopId("rmsnorm"));
for (let i = 0; i < 3; i++) sp("worker:spark:cpu", "compile", `nvcc rmsnorm#${6 + i}`, `14:28:${30 + i * 5}`, null, loopId("rmsnorm"), { job_id: ulid(`job:rms-run:${i}`) });

// The gemm T1's latest `kta check` is queued behind the exclusive lease (shared jobs pause).
sp("worker:spark:gpu-shared", "job_queue", "queue check gemm@2", "14:27:10", null, loopId("gemm"), { waits_for: "exclusive lease released" });

// You.
sp("human", "human_wait", "inbox: T3 swizzle escalated", "13:36:00", null, ulid("trace:human"), { inbox: "escalated task" });
sp("human", "human_wait", "inbox: vadd result 96% of bound", "13:57:30", null, ulid("trace:human"), { inbox: "suspicious result" });

export const ALL_SPANS = SPANS.sort((a, b) => a.start_ts.localeCompare(b.start_ts));

// Records the tasks produced or were rejected with.
export const REJECTION_T3 = put("check", "t3-swizzle-reject-2", { passed: false, failures: [{ check: "validate_template", error: "undeclared name hw.ldmatrix_x4" }] },
  { actor: tool("kta validate-template", taskId("t3-cutlass-swizzle")), created_at: at("13:35:40"), status: "reproduced" });
export const SUBMISSION_SOFTMAX = put("submission", "t1-softmax", { files: ["family.yaml", "chain.yaml", "gen.py", "README.md"], note: "Added online_softmax step; cols up to 8192 legal." },
  { actor: agent(taskId("t1-softmax")), created_at: at("14:04:00") });
