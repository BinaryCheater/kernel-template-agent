import type { MachineDef } from "../data/types";

export type Tone = "running" | "waiting" | "ok" | "fail" | "human" | "idle";

/** Duration in seconds → "1h 04m", "4m 12s", "38s". */
export function dur(s: number): string {
  if (!Number.isFinite(s)) return "—";
  s = Math.max(0, Math.round(s));
  if (s >= 3600) return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
  if (s >= 60) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
  return `${s}s`;
}

export const since = (iso: string, nowMs: number) => (nowMs - Date.parse(iso)) / 1000;
export const between = (a: string, b: string | null, nowMs: number) => ((b ? Date.parse(b) : nowMs) - Date.parse(a)) / 1000;

/** UTC clock time, "14:07" or "14:07:30". */
export function hhmm(iso: string, seconds = false): string {
  const d = new Date(iso);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}${seconds ? `:${p(d.getUTCSeconds())}` : ""}`;
}

export const shortId = (id: string) => (id.length > 26 ? id.slice(0, 10) : id.slice(-8));
export const us = (v: number | null | undefined) => (v == null ? "—" : `${v.toFixed(1)} µs`);
export const pct = (v: number | null | undefined, digits = 0) => (v == null ? "—" : `${(v * 100).toFixed(digits)}%`);
export const fix1 = (v: number | null | undefined) => (v == null ? "—" : v.toFixed(1));

const WORD_TONE: Record<string, Tone> = {
  running: "running", leased: "running", submitted: "running", checking: "running",
  queued: "waiting", waiting: "waiting", created: "waiting", gap: "waiting", analyzing: "waiting",
  accepted: "ok", covered: "ok", ok: "ok", passed: "ok", reproduced: "ok", calibrated: "ok", holdout_validated: "ok", healthy: "ok",
  rejected: "fail", fail: "fail", failed: "fail", expired: "fail", down: "fail", suspect: "fail",
  escalated: "human", proposed: "idle", cancelled: "idle", "not started": "idle", stale: "idle", deprecated: "idle", degraded: "waiting",
};

/** Tone of a loop or task state, from the machine definition: wait → waiting, terminal → ok. */
export function stateTone(m: MachineDef | undefined, state: string): Tone {
  if (m?.id === "task") return WORD_TONE[state] ?? "running";
  const def = m?.states.find((s) => s.id === state);
  if (!def) return WORD_TONE[state] ?? "idle";
  if (def.kind === "wait") return "waiting";
  if (def.kind === "terminal") return "ok";
  return "running";
}

export const wordTone = (w: string | null | undefined): Tone => (w ? WORD_TONE[w] ?? "idle" : "idle");

/** Refinement levels 1-7 of the research frontend (D441); records stored with the paper's 0-5
 *  are upcast by the API before they reach the page. */
export const LEVELS = [1, 2, 3, 4, 5, 6, 7];
export const LEVEL_NAMES: Record<number, string> = {
  1: "workload", 2: "graph", 3: "operators", 4: "tensor map", 5: "instructions", 6: "architecture", 7: "microarchitecture",
};
