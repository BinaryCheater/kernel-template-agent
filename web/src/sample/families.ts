// kta:fixture sample data before the store exists (D95); deleted in WP9a phase B.
// Toy Templates and Families (D94: very simple cases are fine).

import type { ChainNode, FamilyDetail } from "../data/types";
import { at } from "./ids";
import { put, ref } from "./records";

interface Tpl {
  name: string;
  version: number;
  from: number;
  to: number;
  change: string;
  refines: string;
  assurance: string;
  summary: string;
  hw?: string[];
  vars: { name: string; domain: string; binding: string }[];
}

const TEMPLATES: Tpl[] = [
  { name: "grid_stride_map", version: 1, from: 1, to: 2, change: "vertical", refines: "map", assurance: "tested",
    summary: "Map each element to a thread with a grid-stride loop so any n fits a fixed grid.",
    vars: [{ name: "block", domain: "{128,256,512}", binding: "compile" }, { name: "unroll", domain: "{1,2,4}", binding: "compile" }] },
  { name: "vector_load", version: 2, from: 2, to: 3, change: "vertical", refines: "transfer", assurance: "tested",
    summary: "Load contiguous elements with 64- or 128-bit global loads when alignment allows.", hw: ["ldg_128"],
    vars: [{ name: "vec", domain: "{1,2,4,8}", binding: "compile" }] },
  { name: "coalesced_store", version: 1, from: 2, to: 3, change: "vertical", refines: "store", assurance: "tested",
    summary: "Write results so consecutive threads write consecutive addresses.",
    vars: [{ name: "vec", domain: "{1,2,4}", binding: "compile" }] },
  { name: "row_per_cta", version: 1, from: 1, to: 2, change: "vertical", refines: "map", assurance: "tested",
    summary: "Assign one row of a row-wise operator to one CTA.",
    vars: [{ name: "threads", domain: "{128,256,512,1024}", binding: "compile" }] },
  { name: "online_softmax", version: 1, from: 1, to: 1, change: "horizontal", refines: "reduce", assurance: "bounded",
    summary: "Fuse max and sum into one pass using the rescalable property of the exponential sum.",
    vars: [] },
  { name: "warp_reduce", version: 1, from: 2, to: 3, change: "vertical", refines: "reduce", assurance: "tested",
    summary: "Reduce within a warp with shuffle instructions (log2(32) steps).", hw: ["shfl_sync"],
    vars: [] },
  { name: "block_reduce_smem", version: 1, from: 2, to: 3, change: "vertical", refines: "reduce", assurance: "tested",
    summary: "Combine per-warp partials through shared memory, then one warp finishes.",
    vars: [{ name: "slots", domain: "threads/32", binding: "compile" }] },
  { name: "cta_tile", version: 1, from: 1, to: 2, change: "vertical", refines: "contract", assurance: "tested",
    summary: "Tile the output into BM x BN blocks per CTA and loop over K in BK steps.",
    vars: [{ name: "BM", domain: "{32,64,128}", binding: "compile" }, { name: "BN", domain: "{32,64,128}", binding: "compile" }, { name: "BK", domain: "{16,32,64}", binding: "compile" }] },
  { name: "warp_tile", version: 1, from: 2, to: 3, change: "vertical", refines: "contract", assurance: "tested",
    summary: "Split the CTA tile among warps; each warp owns a WM x WN sub-tile.",
    vars: [{ name: "WM", domain: "{16,32,64}", binding: "compile" }, { name: "WN", domain: "{16,32,64}", binding: "compile" }] },
  { name: "mma_bf16", version: 1, from: 3, to: 5, change: "vertical", refines: "contract", assurance: "tested",
    summary: "Issue mma.sync m16n8k16 bf16 with fp32 accumulation on the warp tile.", hw: ["mma_sync_bf16"],
    vars: [] },
  { name: "epilogue_store", version: 1, from: 2, to: 3, change: "vertical", refines: "store", assurance: "tested",
    summary: "Convert accumulators and store the C tile with vectorized writes.",
    vars: [{ name: "vec", domain: "{2,4,8}", binding: "compile" }] },
  { name: "smem_double_buffer", version: 1, from: 3, to: 3, change: "expansion", refines: "transfer", assurance: "declared",
    summary: "Overlap the next K-tile's global-to-shared copy (cp.async) with the current tile's math.", hw: ["cp_async"],
    vars: [{ name: "stages", domain: "{2,3}", binding: "compile" }] },
  { name: "rms_scale", version: 1, from: 1, to: 2, change: "horizontal", refines: "map", assurance: "declared",
    summary: "Fuse the 1/rms scaling and the weight multiply into the store pass.",
    vars: [] },
];

for (const t of TEMPLATES) {
  put("template", `${t.name}@${t.version}`, {
    name: t.name, version: t.version, level_from: t.from, level_to: t.to, change: t.change,
    refines: t.refines, summary: t.summary, requires_hw: t.hw ?? [], variables: t.vars,
  }, { status: t.assurance === "declared" ? "proposed" : "reproduced", assurance: t.assurance });
}

const tplByName = new Map(TEMPLATES.map((t) => [`${t.name}@${t.version}`, t]));

type Item = [key: string, template: string, appliesTo: string, after: string[], bindings?: Record<string, string | number>];

interface FamSpec {
  name: string;
  version: number;
  backend: string;
  problem: string;
  workload: string;
  items: Item[];
  variables: FamilyDetail["variables"];
  glue: FamilyDetail["glue"];
  versions: { version: number; note: string; created_at: string }[];
}

const FAMILIES: FamSpec[] = [
  { name: "vadd_f32", version: 1, backend: "cuda", problem: "vadd_f32", workload: "c[i] = a[i] + b[i], fp32",
    items: [
      ["grid", "grid_stride_map@1", "elementwise add over n", [], { block: "block", unroll: "unroll" }],
      ["ld", "vector_load@2", "loads of a and b", ["grid"], { vec: "vec" }],
      ["st", "coalesced_store@1", "store of c", ["grid"], { vec: "vec" }],
    ],
    variables: [
      { name: "block", domain: "{128,256,512}", meaning: "threads per CTA" },
      { name: "unroll", domain: "{1,2,4}", meaning: "elements per thread per iteration / vec" },
      { name: "vec", domain: "{1,2,4}", meaning: "elements per load and store" },
    ],
    glue: [{ id: "g1", expr: "n % vec == 0 or tail_loop", meaning: "Tail handling spans load and store steps, so it belongs to no single step" }],
    versions: [{ version: 1, note: "initial family from T1", created_at: at("11:02") }] },
  { name: "softmax_rowwise", version: 2, backend: "tilelang", problem: "softmax_bf16", workload: "y = softmax(x, dim=-1), bf16 rows",
    items: [
      ["row", "row_per_cta@1", "one row per CTA", [], { threads: "threads" }],
      ["online", "online_softmax@1", "max+sum in one pass", ["row"]],
      ["ld", "vector_load@2", "row loads", ["row"], { vec: 8 }],
      ["wred", "warp_reduce@1", "per-warp max and sum", ["online"]],
      ["bred", "block_reduce_smem@1", "cross-warp combine", ["wred"], { slots: "threads/32" }],
      ["st", "coalesced_store@1", "normalized row store", ["bred", "ld"], { vec: 8 }],
    ],
    variables: [
      { name: "threads", domain: "{128,256,512,1024}", meaning: "threads per CTA" },
    ],
    glue: [{ id: "g1", expr: "cols % (threads*8) == 0 or masked", meaning: "Masking couples load and store widths" }],
    versions: [
      { version: 1, note: "two-pass max/sum", created_at: at("10:15") },
      { version: 2, note: "T1: online softmax step added (fewer DRAM passes)", created_at: at("14:05") },
    ] },
  { name: "gemm_tiled", version: 1, backend: "tilelang", problem: "gemm_bf16", workload: "C = A @ B, bf16 in, fp32 accumulate",
    items: [
      ["cta", "cta_tile@1", "C tiling over (M,N), K loop", [], { BM: "BM", BN: "BN", BK: "BK" }],
      ["warp", "warp_tile@1", "warp sub-tiles", ["cta"], { WM: "WM", WN: "WN" }],
      ["a_ld", "vector_load@2", "A tile global→smem", ["cta"], { vec: 8 }],
      ["b_ld", "vector_load@2", "B tile global→smem", ["cta"], { vec: 8 }],
      ["mma", "mma_bf16@1", "warp-tile multiply-accumulate", ["warp", "a_ld", "b_ld"]],
      ["epi", "epilogue_store@1", "C tile store", ["mma"], { vec: 4 }],
    ],
    variables: [
      { name: "BM", domain: "{32,64,128}", meaning: "CTA tile rows" },
      { name: "BN", domain: "{32,64,128}", meaning: "CTA tile cols" },
      { name: "BK", domain: "{16,32,64}", meaning: "K step" },
      { name: "WM", domain: "{16,32,64}", meaning: "warp tile rows" },
      { name: "WN", domain: "{16,32,64}", meaning: "warp tile cols" },
    ],
    glue: [{ id: "g1", expr: "(BM*BK + BK*BN)*2 <= hw.smem_per_cta", meaning: "Shared-memory budget is shared by the A and B load steps" }],
    versions: [{ version: 1, note: "initial five-step chain", created_at: at("12:40") }] },
  { name: "rmsnorm_fused", version: 1, backend: "tilelang", problem: "rmsnorm_bf16", workload: "y = x / rms(x) * w, bf16 rows",
    items: [
      ["row", "row_per_cta@1", "one row per CTA", [], { threads: "threads" }],
      ["ld", "vector_load@2", "row and weight loads", ["row"], { vec: 8 }],
      ["wred", "warp_reduce@1", "sum of squares", ["ld"]],
      ["bred", "block_reduce_smem@1", "cross-warp combine", ["wred"]],
      ["scale", "rms_scale@1", "scale and weight", ["bred"]],
      ["st", "coalesced_store@1", "row store", ["scale"], { vec: 8 }],
    ],
    variables: [{ name: "threads", domain: "{128,256,512}", meaning: "threads per CTA" }],
    glue: [],
    versions: [{ version: 1, note: "imported from seed", created_at: at("12:30") }] },
];

function nodes(f: FamSpec): ChainNode[] {
  return f.items.map(([key, template, appliesTo, after, bindings]) => {
    const t = tplByName.get(template)!;
    return {
      key, template, template_record: ref("template", template), applies_to: appliesTo, after,
      level_from: t.from, level_to: t.to, change: t.change, assurance: t.assurance, summary: t.summary,
      requires_hw: t.hw ?? [], variables: t.vars, bindings: bindings ?? {},
    };
  });
}

export const FAMILY_DETAILS: FamilyDetail[] = FAMILIES.map((f) => {
  const label = `${f.name}@${f.version}`;
  const items = nodes(f);
  const chain = put("chain", label, { workload: f.workload, items: items.map(({ key, template, applies_to, after, bindings }) => ({ key, template, applies_to, after, bindings })) },
    { refs: { templates: [...new Set(items.map((n) => n.template_record))] } });
  const versions = f.versions.map((v) => ({
    ...v,
    record: put("family", `${f.name}@${v.version}`, { name: f.name, version: v.version, backend: f.backend, entry: "gen.py:generate" },
      { refs: { chain }, created_at: v.created_at, status: "reproduced" }),
  }));
  return {
    id: versions[versions.length - 1].record, name: f.name, version: f.version, backend: f.backend, problem: f.problem,
    steps: items.length, solutions: 0, workload: f.workload, chain_record: chain, nodes: items,
    variables: f.variables, glue: f.glue, versions, solution_ids: [],
  };
});

export const familyByName = (name: string) => FAMILY_DETAILS.find((f) => f.name === name)!;
