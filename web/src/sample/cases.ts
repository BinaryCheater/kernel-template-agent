// kta:fixture sample data before the store exists (D95); deleted in WP9a phase B.
// Case details, results and their evidence chains, consistent with world.ts.

import type { CaseDetail, Candidate, ResultDetail, Round, RoundTask } from "../data/types";
import { at } from "./ids";
import { familyByName } from "./families";
import { HWMODEL, put, ENV_SPARK, WORKER_SPARK, checker, tool } from "./records";
import { REJECTION_T3, SUBMISSION_SOFTMAX, TASK_SCRIPTS, loopId, roundSteps, subjectId, taskCard, taskId } from "./world";

interface CaseSpec {
  key: string;
  name: string;
  problem: string;
  shapes: string;
  objective: string;
  bound_us: number;
  bottleneck: string;
  refs: [string, number][];
  buckets: { bucket: string; bound_us: number; best_us?: number; references: Record<string, number> }[];
  best_us?: number;
  family: string;
  candidates: [Record<string, number | string>, number, number?][];
  outcomes: Record<number, { rule?: number; label: string }>;
}

const SPECS: CaseSpec[] = [
  { key: "vadd", name: "vadd_f32", problem: "vadd_f32", shapes: "n ∈ {2^20, 2^24}, fp32", objective: "weighted_latency",
    bound_us: 41.9, bottleneck: "dram", refs: [["torch.add (eager)", 47.8], ["cub::DeviceTransform", 44.9]],
    buckets: [
      { bucket: "n=2^20", bound_us: 2.6, best_us: 4.1, references: { "torch.add (eager)": 6.0, "cub::DeviceTransform": 4.4 } },
      { bucket: "n=2^24", bound_us: 41.9, best_us: 43.6, references: { "torch.add (eager)": 47.8, "cub::DeviceTransform": 44.9 } },
    ],
    best_us: 43.6, family: "vadd_f32",
    candidates: [[{ block: 256, unroll: 2, vec: 4 }, 42.1, 43.6], [{ block: 512, unroll: 1, vec: 4 }, 42.3, 43.9], [{ block: 256, unroll: 1, vec: 4 }, 42.4, 44.0],
      [{ block: 128, unroll: 2, vec: 4 }, 42.6, 44.4], [{ block: 256, unroll: 4, vec: 2 }, 43.0, 45.2], [{ block: 512, unroll: 2, vec: 2 }, 43.1, 45.0]],
    outcomes: { 1: { rule: 3, label: "accepted: beats strongest reference; gap to bound explained (launch + tail)" } } },
  { key: "softmax", name: "softmax_bf16", problem: "softmax_bf16", shapes: "rows 4096 × cols 1024, bf16", objective: "weighted_latency",
    bound_us: 7.9, bottleneck: "dram", refs: [["torch.softmax (eager)", 14.2], ["triton tutorial", 9.8]],
    buckets: [{ bucket: "4096×1024", bound_us: 7.9, references: { "torch.softmax (eager)": 14.2, "triton tutorial": 9.8 } }],
    family: "softmax_rowwise",
    candidates: [[{ threads: 256 }, 8.6], [{ threads: 512 }, 8.7], [{ threads: 128 }, 8.9], [{ threads: 1024 }, 9.3]],
    outcomes: { 1: { label: "no legal Template → T1 extend softmax_rowwise" } } },
  { key: "gemm", name: "gemm_bf16_256", problem: "gemm_bf16", shapes: "M=N=K=256, bf16 → fp32 acc", objective: "weighted_latency",
    bound_us: 3.1, bottleneck: "launch", refs: [["torch.matmul (cuBLAS)", 6.4], ["cutlass sgemm example", 7.9]],
    buckets: [{ bucket: "256³", bound_us: 3.1, best_us: 9.7, references: { "torch.matmul (cuBLAS)": 6.4, "cutlass sgemm example": 7.9 } }],
    best_us: 9.7, family: "gemm_tiled",
    candidates: [[{ BM: 64, BN: 64, BK: 32, WM: 32, WN: 32 }, 6.1, 9.7], [{ BM: 64, BN: 32, BK: 32, WM: 32, WN: 16 }, 6.3, 10.2],
      [{ BM: 32, BN: 64, BK: 32, WM: 16, WN: 32 }, 6.4, 10.4], [{ BM: 128, BN: 64, BK: 32, WM: 64, WN: 32 }, 6.6, 12.8],
      [{ BM: 64, BN: 64, BK: 64, WM: 32, WN: 32 }, 6.7, 11.1], [{ BM: 32, BN: 32, BK: 32, WM: 16, WN: 16 }, 7.0, 10.9]],
    outcomes: { 1: { rule: 4, label: "gap unexplained (38%) → T1 extend gemm_tiled" } } },
  { key: "rmsnorm", name: "rmsnorm_bf16", problem: "rmsnorm_bf16", shapes: "rows 2048 × cols 4096, bf16", objective: "weighted_latency",
    bound_us: 30.7, bottleneck: "dram", refs: [["torch rms_norm (eager)", 61.0]],
    buckets: [{ bucket: "2048×4096", bound_us: 30.7, references: { "torch rms_norm (eager)": 61.0 } }],
    family: "rmsnorm_fused", candidates: [[{ threads: 256 }, 32.2], [{ threads: 512 }, 32.5], [{ threads: 128 }, 33.9]], outcomes: {} },
];

function roundTasks(key: string): RoundTask[] {
  return TASK_SCRIPTS.filter((t) => t.loop === key).map((t) => ({
    id: taskId(t.key), type: t.type, card: taskCard(t), created_at: at(t.created), ended_at: t.ended ? at(t.ended) : null, state: t.state,
    rejections: t.key === "t3-cutlass-swizzle" ? [{ at: at("13:21:00"), reason: "validate_template: variable 'swz' has no domain", record: REJECTION_T3 },
      { at: at("13:35:40"), reason: "validate_template: precondition references undeclared name hw.ldmatrix_x4", record: REJECTION_T3 }] : [],
  }));
}

export const RESULTS: ResultDetail[] = [];
export const CASES: CaseDetail[] = SPECS.map((c) => {
  const caseId = subjectId("solve", c.name);
  const problem = put("problem", c.problem, { name: c.problem, reference_impl: `${c.problem}_ref.py` });
  put("case", c.name, { problem: c.problem, shape_dist: c.shapes, objective: c.objective, target: "gb10" }, { refs: { problem } });
  const bound = put("bound", c.name, { bound_us: c.bound_us, bottleneck: c.bottleneck }, { refs: { case: caseId, hw_model: HWMODEL }, actor: tool("kta bound") });
  const fam = familyByName(c.family);
  const references = c.refs.map(([name, us]) => ({ name, us,
    record: put("measurement", `${c.name}:ref:${name}`, { median_us: us, ci_low_us: us * 0.99, ci_high_us: us * 1.01, n: 50, protocol: "measure-v1" },
      { refs: { env: ENV_SPARK, worker: WORKER_SPARK }, actor: tool("kta measure"), status: "reproduced" }) }));
  const candidates: Candidate[] = c.candidates.map(([assignment, pred, meas], i) => ({
    rank: i + 1, assignment, predicted_us: pred, measured_us: meas, build_ok: true, check_passed: meas !== undefined ? true : undefined,
    record: put("candidate", `${c.name}:${i}`, { assignment, bucket: c.buckets[c.buckets.length - 1].bucket }, { refs: { family: fam.id } }),
  }));
  const steps = roundSteps(c.key);
  const tasks = roundTasks(c.key);
  const rounds: Round[] = steps.map((s, i) => {
    const o = c.outcomes[i + 1];
    return { n: i + 1, steps: s, tasks: tasks.filter((t) => t.created_at >= s[0].start && (!s[s.length - 1].end || t.created_at <= s[s.length - 1].end!)),
      outcome: o ? { ...o, record: put("report", `${c.name}:judgement:${i + 1}`, { kind: "judgement", rule: o.rule ?? null, text: o.label }, { refs: { case: caseId }, actor: { type: "system", name: "orchestrator" } }) } : undefined };
  });

  let best: CaseDetail["best"];
  if (c.best_us !== undefined) {
    const r = buildResult(c, caseId, bound, candidates[0], fam.id);
    RESULTS.push(r);
    best = { us: c.best_us, record: r.chain.find((l) => l.stage === "Measurement")!.record, solution: r.id, family: fam.name };
    if (c.key === "vadd") { fam.solution_ids.push({ id: r.id, label: `${c.name} · ${c.buckets[1].bucket}` }); fam.solutions = 1; }
  }
  const gaps = c.key === "gemm" ? [put("gap", "gemm:latency", { missing: "template", description: "Global-load latency exposed in the K loop; references overlap loads with mma via cp.async pipelining", level: 3, attribution: "space" }, { refs: { case: caseId } })] : [];
  const loop = loopId(c.key);
  const last = steps[steps.length - 1];
  return {
    id: caseId, record: caseId, name: c.name, problem: c.problem, loop_id: loop, state: last[last.length - 1].state, round: steps.length,
    best_us: c.best_us, bound_us: c.bound_us, pct_of_bound: c.best_us ? c.bound_us / c.best_us : undefined, objective: c.objective,
    bound_record: bound, shapes: c.shapes, references, best, rounds,
    buckets: c.buckets.map((b) => ({ ...b, solution: undefined })), candidates, gaps,
  };
});

function buildResult(c: CaseSpec, caseId: string, bound: string, top: Candidate, familyId: string): ResultDetail {
  const fam = familyByName(c.family);
  const isVadd = c.key === "vadd";
  const pred = put("prediction", `${c.name}:top`, { per_unit: isVadd ? { dram: 41.0, launch: 1.1 } : { tensor: 0.9, smem: 1.6, dram: 0.6, launch: 3.0 },
    serial: isVadd ? 1.1 : 3.0, total: top.predicted_us, model: "gb10@2" }, { refs: { candidate: top.record, hw_model: HWMODEL }, actor: tool("kta solve") });
  const build = put("build", `${c.name}:top`, { ok: true, regs: isVadd ? 18 : 96, spills: 0, smem: isVadd ? 0 : 16384, log_summary: "ptxas: 0 bytes spill" },
    { refs: { candidate: top.record, env: ENV_SPARK, worker: WORKER_SPARK }, actor: tool("kta build") });
  const sass = put("check", `${c.name}:sass`, { passed: true, failures: [], expected: isVadd ? ["LDG.E.128", "STG.E.128"] : ["HMMA.16816.F32.BF16", "LDS.128"] },
    { refs: { build }, actor: checker("check_sass@1"), status: "reproduced" });
  const check = put("check", `${c.name}:numeric`, { passed: true, failures: [], distributions: ["normal", "uniform_wide", "large_magnitude", "denormal", "zeros", "edge_shapes"] },
    { refs: { build, env: ENV_SPARK, worker: WORKER_SPARK }, actor: checker("numeric_checker@3"), status: "reproduced" });
  const meas = put("measurement", `${c.name}:top`, { median_us: c.best_us, ci_low_us: c.best_us! - 0.3, ci_high_us: c.best_us! + 0.4, n: 200, protocol: "measure-v1", clocks: { sm_mhz: 2418 }, temps: { gpu_c: 51 } },
    { refs: { build, env: ENV_SPARK, worker: WORKER_SPARK }, actor: tool("kta measure"), status: "reproduced" });
  const sol = put("solution", `${c.name}:best`, { bucket: c.buckets[c.buckets.length - 1].bucket, assignment: top.assignment, pct_of_bound: c.bound_us / c.best_us! },
    { refs: { case: caseId, family: familyId, measurement: meas, check, bound }, actor: { type: "system", name: "orchestrator" }, created_at: at(isVadd ? "13:57:30" : "14:08:30"),
      status: isVadd ? "reproduced" : "proposed" });
  const templates = fam.nodes.map((n) => n.template).join(" → ");
  return {
    id: sol, case_id: caseId, case_name: c.name, family: familyId, family_name: `${fam.name}@${fam.version}`, bucket: c.buckets[c.buckets.length - 1].bucket,
    measured_us: c.best_us!, bound_us: c.bound_us, pct_of_bound: c.bound_us / c.best_us!,
    verdict: isVadd ? "Accepted by rule 3: faster than the strongest reference (44.9 µs) and the remaining 1.7 µs is explained by launch overhead and the tail iteration."
      : "Not accepted: rule 4 fired. 38% of the gap to the bound is not explained by the model; a T1 task is extending the family.",
    chain: [
      { stage: "Templates", title: `${fam.name}@${fam.version} chain`, fact: templates, record: fam.chain_record, status: "tested" },
      { stage: "Assignment", title: "Chosen candidate", fact: Object.entries(top.assignment).map(([k, v]) => `${k}=${v}`).join(", "), record: top.record },
      { stage: "Prediction", title: "Pre-compile model gb10@2", fact: `${top.predicted_us} µs predicted`, record: pred },
      { stage: "Build", title: "nvcc 13.0.88 on spark", fact: isVadd ? "18 regs, 0 spills" : "96 regs, 0 spills, 16 KiB smem", record: build, status: "ok" },
      { stage: "SASS check", title: "check_sass@1", fact: `expected ${isVadd ? "LDG.E.128, STG.E.128" : "HMMA.16816.F32.BF16, LDS.128"} present`, record: sass, status: "passed" },
      { stage: "Correctness", title: "numeric_checker@3", fact: "passed 6/6 input distributions incl. edge shapes", record: check, status: "passed" },
      { stage: "Measurement", title: "measure-v1, n=200, spark-gpu0", fact: `median ${c.best_us} µs [${(c.best_us! - 0.3).toFixed(1)}, ${(c.best_us! + 0.4).toFixed(1)}]`, record: meas, status: "reproduced" },
      { stage: "Gap to bound", title: `bound ${c.bound_us} µs (${c.bottleneck})`, fact: `${((c.bound_us / c.best_us!) * 100).toFixed(0)}% of bound`, record: bound },
    ],
    units: isVadd
      ? [{ unit: "dram", bound_us: 41.9, predicted_us: 41.0, measured_us: 42.3 }, { unit: "launch", bound_us: 0, predicted_us: 1.1, measured_us: 1.3 }]
      : [{ unit: "launch", bound_us: 3.1, predicted_us: 3.0, measured_us: 3.2 }, { unit: "smem", bound_us: 0.4, predicted_us: 1.6, measured_us: 3.9 },
        { unit: "tensor", bound_us: 0.3, predicted_us: 0.9, measured_us: 1.0 }, { unit: "dram", bound_us: 0.2, predicted_us: 0.6, measured_us: 1.6 }],
    explained_share: isVadd ? 0.88 : 0.62,
  };
}

// Mark the softmax submission as referenced so it is reachable from the case page.
export const SOFTMAX_SUBMISSION = SUBMISSION_SOFTMAX;
