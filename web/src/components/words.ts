// Plain words for the system's internal names (WP25, D669). Display only: ids stay in
// records and links. A name not listed here is shown as it is.

/** Loop states → what a person calls that step. */
export const STATE_WORDS: Record<string, string> = {
  // solve loop: one Case
  new: "Start",
  bounded: "Lower bound",
  retrieving: "Formulation",
  solving: "Compile + solve",
  building: "Compile",
  checking: "Check",
  measuring: "Measure",
  judging: "Judge",
  waiting: "Agent formulates",
  accepted: "Solved",
  closed: "Gave up",
  // expand loop: one Reference
  frozen: "Pin source",
  measured: "Measure it",
  analyzing: "Agent decomposes",
  gap: "Agent extracts",
  reconstructed: "Rebuild",
  reproduced: "Reproduce",
  ablated: "Ablate",
  covered: "Learned",
};

export const stateWord = (s: string) => STATE_WORDS[s] ?? s;

/** Agent task types → what the agent is asked to do. */
export const TASK_WORDS: Record<string, string> = {
  T1: "build a kernel family",
  T2: "decompose a reference",
  T3: "extract a template",
  T4: "fix the cost model",
  T5: "write a microbenchmark",
  T6: "fix a family or template",
  T7: "analyse the round",
};

export const taskWord = (t: string) => TASK_WORDS[t] ?? t;

/** The seven optimisation levels (D441), short names for narrow places. */
export const LEVEL_SHORT: Record<number, string> = {
  1: "workload", 2: "graph", 3: "operators", 4: "tensor map", 5: "instructions", 6: "architecture", 7: "micro-arch",
};

/** Where Templates were learned from: Source and Reference names → a short name. */
const SOURCE_SHORT: Record<string, string> = {
  learning_to_formulate_paper: "Paper",
  cutlass_ampere_tf32_gemm: "CUTLASS GEMM",
  flash_attention_2_fwd: "FlashAttention-2",
  flash_attention_3_fwd: "FlashAttention-3",
  marlin_int4_gemm: "Marlin INT4 GEMM",
  flashinfer_batch_decode: "FlashInfer decode",
  vllm_fused_moe: "vLLM fused MoE",
  vllm_rms_norm: "vLLM RMSNorm",
};

export const sourceWord = (s: string) => {
  const k = s.replace(/_source$/, "");
  return SOURCE_SHORT[k] ?? k.replace(/_/g, " ");
};
