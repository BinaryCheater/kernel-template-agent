// One async function per API endpoint (spec/modules/api-web.md). The Vite dev server proxies
// /api to the KTA API on the control host (config [api]); nothing here holds data itself.

import type {
  ActiveRuns, CaseDetail, CaseIndexRow, CoverageReport, FamilyDetail, FamilySummary, InboxItem,
  Machines, Meta, RecordEnvelope, ResultDetail, Snapshot, Span, TraceInfo,
} from "./types";

const BASE = "/api";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function get<T>(path: string, params: Record<string, string | undefined> = {}): Promise<T> {
  const q = new URLSearchParams(Object.entries(params).filter((e): e is [string, string] => e[1] !== undefined && e[1] !== ""));
  const res = await fetch(`${BASE}${path}${q.size ? `?${q}` : ""}`);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: string };
      detail = body.detail ?? detail;
    } catch {
      // A non-JSON error body: keep the status text.
    }
    throw new ApiError(res.status, `${res.status} ${detail}`);
  }
  return (await res.json()) as T;
}

const enc = encodeURIComponent;

export const api = {
  meta: () => get<Meta>("/meta"),
  machines: () => get<Machines>("/machines"),
  active: () => get<ActiveRuns>("/runs/active"),
  inbox: () => get<InboxItem[]>("/inbox"),
  trace: (from: number, to: number) =>
    get<{ spans: Span[]; traces: TraceInfo[] }>("/trace", { from: new Date(from).toISOString(), to: new Date(to).toISOString() }),
  record: (id: string) => get<RecordEnvelope>(`/records/${enc(id)}`),
  cases: () => get<CaseIndexRow[]>("/cases"),
  caseTimeline: (id: string) => get<CaseDetail>(`/cases/${enc(id)}/timeline`),
  lineage: async (id: string) => {
    const g = await get<{ result?: ResultDetail }>(`/lineage/${enc(id)}`);
    if (!g.result) throw new ApiError(400, "this record is not a Solution; the evidence chain is shown for Solutions");
    return g.result;
  },
  families: (snapshot?: string) => get<FamilySummary[]>("/families", { snapshot }),
  familyChain: (id: string) => get<FamilyDetail>(`/families/${enc(id)}/chain`),
  snapshots: () => get<Snapshot[]>("/snapshots"),
  coverage: (snapshot: string) => get<CoverageReport>("/coverage", { snapshot }),
};

/** Wall-clock "now" for elapsed times. */
export const nowMs = (): number => Date.now();
