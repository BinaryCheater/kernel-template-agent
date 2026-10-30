// kta:fixture sample data before the store exists (D95); deleted in WP9a phase B.
// Every sample record carries origin "fixture" so it can never pass for evidence.

import type { RecordEnvelope } from "../data/types";
import { at, rid } from "./ids";

export const RECORDS = new Map<string, RecordEnvelope>();

interface PutOpts {
  refs?: Record<string, string | string[]>;
  actor?: RecordEnvelope["actor"];
  created_at?: string;
  status?: string;
  assurance?: string;
  blobs?: Record<string, string>;
}

/** Register a sample record under a stable label and return its id. */
export function put(kind: string, label: string, body: Record<string, unknown>, o: PutOpts = {}): string {
  const id = rid(`${kind}:${label}`);
  RECORDS.set(id, {
    id,
    kind,
    schema: 1,
    created_at: o.created_at ?? at("09:00"),
    actor: o.actor ?? { type: "system", name: "sample" },
    refs: o.refs ?? {},
    body,
    blobs: o.blobs ?? {},
    origin: "fixture",
    status: o.status ?? "proposed",
    assurance: o.assurance,
  });
  return id;
}

/** Look up a previously registered record id by kind and label. */
export function ref(kind: string, label: string): string {
  const id = rid(`${kind}:${label}`);
  if (!RECORDS.has(id)) throw new Error(`sample record missing: ${kind}:${label}`);
  return id;
}

export const tool = (name: string, task?: string) => ({ type: "tool", name, task });
export const checker = (name: string) => ({ type: "checker", name });
export const agent = (task: string) => ({ type: "agent", name: "worker", task });

// ---- shared records: target, workers, environment ----

export const ENV_CONTROL = put("env_manifest", "control", {
  role: "control", host: "mac-control", os: "Darwin 27.2", arch: "arm64", python: "3.12.13",
  uv: "0.11.28", node: "26.12.1", commit: "8819762", dirty: false, lock_mismatches: [],
});
export const ENV_SPARK = put("env_manifest", "spark", {
  role: "worker", host: "spark", os: "Linux 6.11", arch: "aarch64", python: "3.12.13",
  cuda: "13.0.3", driver: "580.95.05", nvcc: "13.0.88", ptxas: "13.0.88", commit: "8819762",
  dirty: false, lock_mismatches: [],
});
export const ENV_SLAI = put("env_manifest", "slai", {
  role: "worker", host: "slai", os: "Linux 6.8", arch: "x86_64", python: "3.12.13",
  commit: "8819762", dirty: false, lock_mismatches: [],
});
export const TARGET = put("target", "gb10", {
  name: "gb10", gpu: "NVIDIA GB10", sm: "sm_121", driver: "580.95.05", cuda: "13.0.3",
  ptxas: "13.0.88", backends: { tilelang: "0.1.6", cuda: "13.0" },
});
export const WORKER_SPARK = put("worker", "spark-gpu0", {
  name: "spark-gpu0", host: "spark", transport: "ssh",
  labels: { gpu: "sm_121", cuda: "13.0", pool: "gpu-exclusive,gpu-shared,cpu" },
  devices: ["0"], commit: "8819762",
}, { refs: { env: ENV_SPARK } });
export const WORKER_SLAI = put("worker", "slai-cpu", {
  name: "slai-cpu", host: "slai", transport: "ssh", labels: { pool: "cpu" }, devices: [],
  commit: "8819762",
}, { refs: { env: ENV_SLAI } });
export const HWMODEL = put("hw_model", "gb10@2", {
  target: "gb10", version: 2,
  peaks: { dram: 273e9, l2: 1.1e12, smem: 3.2e12, tensor: 1.0e15, fma: 31e12 },
  limits: { smem_per_cta: 101376, regs_per_thread: 255, max_threads_per_cta: 1024 },
}, { refs: { target: TARGET }, status: "calibrated" });
