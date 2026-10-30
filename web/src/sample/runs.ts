// kta:fixture sample data before the store exists (D95); deleted in WP9a phase B.
// Active runs, inbox, pool snapshots and coverage reports.

import type { ActiveRuns, CoverageReport, InboxItem, Snapshot } from "../data/types";
import { AS_OF, at, ulid } from "./ids";
import { WORKER_SLAI, WORKER_SPARK, put } from "./records";
import { RESULTS } from "./cases";
import { REJECTION_T3, buildLoops, buildTasks, loopId, taskId } from "./world";

export function activeRuns(): ActiveRuns {
  const loops = buildLoops();
  const tasks = buildTasks();
  const compile = (n: number, state: "queued" | "running", start: string) =>
    Array.from({ length: n }, (_, i) => ({
      id: ulid(`job:rms-${state}:${i}`), kind: "compile", name: `nvcc rmsnorm#${(state === "running" ? 6 : 9) + i}`, pool: "cpu" as const,
      worker: "spark-gpu0", state, queued_at: at(start), started_at: state === "running" ? at(`14:28:${30 + i * 5}`) : undefined, loop_id: loopId("rmsnorm"),
    }));
  return {
    as_of: AS_OF,
    agent_slots: ["agent:1", "agent:2"],
    loops,
    tasks,
    jobs: [
      { id: ulid("job:measure-softmax"), kind: "measure", name: "measure softmax top8", pool: "gpu-exclusive", worker: "spark-gpu0", state: "running",
        queued_at: at("14:19:00"), started_at: at("14:27:00"), loop_id: loopId("softmax") },
      { id: ulid("job:check-gemm2"), kind: "check", name: "check gemm@2", pool: "gpu-shared", worker: "spark-gpu0", state: "queued",
        queued_at: at("14:27:10"), task_id: taskId("t1-gemm") },
      ...compile(3, "running", "14:22:00"),
      ...compile(2, "queued", "14:22:00"),
    ],
    leases: [{ gpu: "spark-gpu0:0", holder_job: ulid("job:measure-softmax"), since: at("14:27:00"), max_s: 600 }],
    workers: [
      { name: "spark-gpu0", host: "spark", transport: "ssh", labels: { gpu: "sm_121", cuda: "13.0", backend: "tilelang" }, commit: "8819762",
        last_seen: at("14:29:56"), health: "ok", record: WORKER_SPARK,
        pools: [{ pool: "gpu-exclusive", capacity: 1, running: 1, queued: 0 }, { pool: "gpu-shared", capacity: 2, running: 0, queued: 1 },
          { pool: "cpu", capacity: 18, running: 3, queued: 2 }] },
      { name: "slai-cpu", host: "slai", transport: "ssh", labels: { pool: "cpu", arch: "x86_64" }, commit: "8819762",
        last_seen: at("14:29:51"), health: "ok", record: WORKER_SLAI, pools: [{ pool: "cpu", capacity: 30, running: 0, queued: 0 }] },
    ],
    budgets: [
      { key: "global", label: "Agent time today", unit: "min", limit: 480, used: 176, reserved: 105, children: [
        { key: "process:experiment", label: "experiment (50%)", unit: "min", limit: 0, used: 0, reserved: 0 },
        { key: "process:solve", label: "solve", unit: "min", limit: 240, used: 47, reserved: 45 },
        { key: "process:learn", label: "learn", unit: "min", limit: 144, used: 129, reserved: 60 },
        { key: "process:calibrate", label: "calibrate + maintain", unit: "min", limit: 96, used: 0, reserved: 0 },
      ] },
      { key: "global:gpu", label: "GPU time today", unit: "min", limit: 120, used: 41, reserved: 10, children: [
        { key: "process:solve", label: "solve", unit: "min", limit: 60, used: 27, reserved: 10 },
        { key: "process:learn", label: "learn", unit: "min", limit: 36, used: 14, reserved: 0 },
      ] },
    ],
    inbox_count: INBOX.length,
    recently_finished: [
      { id: loopId("cub_reduce"), label: "Reference cub_block_reduce → covered", status: "ok", ended_at: at("14:27:00"), link: "/coverage" },
      { id: taskId("t1-softmax"), label: "T1 extend softmax_rowwise → accepted", status: "ok", ended_at: at("14:07:00"), link: "/pipeline" },
      { id: loopId("vadd"), label: "Case vadd_f32 → accepted (96% of bound)", status: "ok", ended_at: at("13:57:30"), link: `/results/${RESULTS[0].id}` },
      { id: taskId("t3-cutlass-swizzle"), label: "T3 swizzled smem layout → escalated", status: "fail", ended_at: at("13:36:00"), link: "/pipeline" },
    ],
  };
}

export const INBOX: InboxItem[] = [
  { id: ulid("inbox:vadd"), trigger: "Result above 95% of the bound", subject: "Case vadd_f32 · n=2^24 at 96% of bound",
    suggestion: "Re-verify: re-measure on a cold GPU and re-run the numeric checker", record: RESULTS[0]?.id ?? "", at: at("13:57:30") },
  { id: ulid("inbox:t3"), trigger: "Task over retry limit", subject: "T3 swizzled smem layout (cutlass_sgemm_example)",
    suggestion: "Last rejection: precondition references undeclared name hw.ldmatrix_x4 — add the HwFact or narrow the precondition, then retry with a note",
    record: REJECTION_T3, at: at("13:36:00") },
];

export const SNAPSHOTS: Snapshot[] = [
  { id: put("pool_snapshot", "pool@1", { members: [], note: "seed import" }, { created_at: at("09:10") }), label: "pool@1 · seed", created_at: at("09:10"), members: 9 },
  { id: put("pool_snapshot", "pool@2", { members: [], note: "after cub_block_reduce analysis" }, { created_at: at("11:58") }), label: "pool@2", created_at: at("11:58"), members: 11 },
  { id: put("pool_snapshot", "pool@3", { members: [], note: "after softmax T1 and cutlass T3" }, { created_at: at("14:07") }), label: "pool@3 · current", created_at: at("14:07"), members: 13 },
];

const MEAN = [5, 7.6, 9.4, 10.6, 11.5, 12.1, 12.6, 13.0, 13.2];
const SPREAD = [0, 1.1, 1.4, 1.5, 1.4, 1.2, 1.0, 0.7, 0];

export function coverage(snapshotId: string): CoverageReport {
  const idx = Math.max(0, SNAPSHOTS.findIndex((s) => s.id === snapshotId));
  const n = [5, 7, 9][idx];
  const sources = Array.from({ length: n }, (_, i) => i + 1);
  const refs: CoverageReport["references"] = [
    { name: "cub_block_reduce", state: idx >= 1 ? (idx === 2 ? "covered" : "reproduced") : "frozen", record: put("reference", "cub_block_reduce", { name: "cub_block_reduce" }), gaps: 0 },
    { name: "cutlass_sgemm_example", state: idx >= 1 ? "gap" : "frozen", record: put("reference", "cutlass_sgemm_example", { name: "cutlass_sgemm_example" }), gaps: idx >= 1 ? 3 : 0 },
    { name: "triton_softmax_tutorial", state: idx === 2 ? "analyzing" : "not started", record: put("reference", "triton_softmax_tutorial", { name: "triton_softmax_tutorial" }), gaps: 0 },
    { name: "flashinfer_decode", state: "not started", record: put("reference", "flashinfer_decode", { name: "flashinfer_decode" }), gaps: 0 },
  ];
  const cells: CoverageReport["hw_patterns"]["cells"] = [];
  const grid = [
    [2, 1, 0, 3, 2, 0], [0, 0, 0, idx === 2 ? 1 : 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 2, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0],
  ];
  const best = [["tested", "tested", "", "tested", "tested", ""], ["", "", "", "declared", "", ""], ["", "", "", "", "", ""], ["", "", "tested", "", "", ""], ["", "bounded", "", "", "", "tested"], ["", "", "", "", "", ""]];
  grid.forEach((row, f) => row.forEach((v, p) => cells.push({ f, p, n: v, best: best[f][p] })));
  return {
    snapshot: snapshotId,
    record: put("coverage_report", `cov:${idx}`, { snapshot: snapshotId }),
    scope: { description: "public kernel tutorials and libraries (toy sample)", total: 9, by_category: { tutorials: 4, libraries: 3, "LLM inference kernels": 2 } },
    discovery: {
      sources, mean: MEAN.slice(0, n), lo: MEAN.slice(0, n).map((m, i) => m - SPREAD[i]), hi: MEAN.slice(0, n).map((m, i) => m + SPREAD[i]),
      chao1: [{ estimate: 11, lo: 9, hi: 19 }, { estimate: 15, lo: 12, hi: 24 }, { estimate: 16, lo: 13.5, hi: 23 }][idx],
    },
    trend: SNAPSHOTS.slice(0, idx + 1).map((s, i) => ({ snapshot: s.label.split(" ")[0], good_turing: [0.44, 0.61, 0.72][i], expressible: [0.25, 0.5, 0.67][i] })),
    instruction_families: [
      { family: "LDG/STG.128 vectorized global access", we_produce: true, sota_uses: true, note: "vector_load@2, coalesced_store@1" },
      { family: "SHFL warp shuffle", we_produce: true, sota_uses: true, note: "warp_reduce@1" },
      { family: "HMMA bf16 (mma.sync m16n8k16)", we_produce: true, sota_uses: true, note: "mma_bf16@1" },
      { family: "LDGSTS (cp.async)", we_produce: idx === 2, sota_uses: true, note: idx === 2 ? "smem_double_buffer@1 (declared, untested)" : "no Template" },
      { family: "LDSM (ldmatrix)", we_produce: false, sota_uses: true, note: "T3 escalated" },
      { family: "UTMALDG (TMA bulk copy)", we_produce: false, sota_uses: true, note: "no Template; feature support on sm_121 to probe" },
      { family: "RED/ATOM global atomics", we_produce: false, sota_uses: false, note: "not seen in sources yet" },
    ],
    hw_patterns: { features: ["ldg_128", "cp_async", "tma", "mma_sync_bf16", "shfl_sync", "atomics"], patterns: ["map", "reduce", "contract", "transfer", "store", "sync"], cells },
    references: refs,
    gap_explained: [
      { case: "vadd_f32", explained: 0.88, unexplained: 0.12 },
      { case: "gemm_bf16_256", explained: 0.62, unexplained: 0.38 },
    ],
    backlog: [
      { item: "flashinfer_decode", why: "Uses cp.async + ldmatrix, both red rows; covers decode attention", priority: 1 },
      { item: "T3 ldmatrix swizzle (escalated)", why: "Blocks cutlass_sgemm_example; needed by gemm_bf16_256 gap", priority: 2 },
      { item: "TMA probe (HwFeature)", why: "Unknown support on sm_121 blocks the TMA row", priority: 3 },
    ],
    structure: { other_patterns: 1, unmatched_by_kind: { layout: 2, epilogue: 1, other: 1 }, assurance: { declared: 2, tested: 10, bounded: 1, proved: 0 } },
  };
}
