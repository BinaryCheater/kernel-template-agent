// One async function per API endpoint (spec/modules/api-web.md). The Vite dev server proxies
// /api to the KTA API on the control host (config [api]); nothing here holds data itself.

import type {
  ActionLogEntry, ActiveRuns, AgentsView, BlobText, Capability, CaseDetail, CaseIndexRow, CoverageReport,
  FamilyDetail, FamilySummary, ImprovementsView, InboxItem, Machines, Meta, PoolView, RecordEnvelope,
  Residual, ResultDetail, Snapshot, Span, TaskDetail, TraceInfo, ExperimentProtocol,
} from "./types";

const BASE = "/api";

export class ApiError extends Error {
  /** `detail` is the server's explanation alone (for 501: what is not wired yet). */
  constructor(public status: number, message: string, public detail: string = message) {
    super(message);
  }
}

async function get<T>(path: string, params: Record<string, string | undefined> = {}): Promise<T> {
  const q = new URLSearchParams(Object.entries(params).filter((e): e is [string, string] => e[1] !== undefined && e[1] !== ""));
  return read<T>(await fetch(`${BASE}${path}${q.size ? `?${q}` : ""}`));
}

/** POST a JSON body (operations). Errors keep their HTTP status: 400 bad request, 404 unknown
 *  record or task, 501 not wired yet (the detail names what is missing). */
async function post<T>(path: string, body: unknown = {}): Promise<T> {
  return read<T>(await fetch(`${BASE}${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }));
}

async function read<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: string };
      detail = body.detail ?? detail;
    } catch {
      // A non-JSON error body: keep the status text.
    }
    throw new ApiError(res.status, `${res.status} ${detail}`, detail);
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
  // WP9b
  capabilities: () => get<Capability[]>("/actions/capabilities"),
  actionLog: (limit = 100) => get<ActionLogEntry[]>("/actions/log", { limit: String(limit) }),
  task: (id: string) => get<TaskDetail>(`/tasks/${enc(id)}`),
  blob: (sha: string) => get<BlobText>(`/blobs/${enc(sha)}`),
  agents: () => get<AgentsView>("/agents"),
  pool: (snapshot?: string) => get<PoolView>("/pool", { snapshot }),
  experiments: async () => (await get<{ protocols: ExperimentProtocol[] }>("/experiments")).protocols,
  improvements: () => get<ImprovementsView>("/improvements"),
  residuals: (model?: string) => get<Residual[]>("/model/residuals", { model }),
  verifyBatch: (batch: string) => post<Record<string, unknown>>(`/agents/batches/${enc(batch)}/verify`),
};

/** Operation name (GET /actions/capabilities) -> endpoint call. Bodies follow spec/modules/api-web.md. */
export const act = {
  reject: (record: string, reason: string) => post<Record<string, unknown>>("/actions/reject", { record, reason }),
  deprecate: (record: string, reason: string) => post<Record<string, unknown>>("/actions/deprecate", { record, reason }),
  flag: (record: string, reason: string) => post<Record<string, unknown>>("/actions/flag", { record, reason }),
  approve: (what: "protocol" | "improvement" | "profile", record: string, reason: string) =>
    post<Record<string, unknown>>("/actions/approve", { [what]: record, reason }),
  confirm: (what: "pattern" | "structure", record: string, reason: string) =>
    post<Record<string, unknown>>("/actions/confirm", { [what]: record, reason }),
  budget: (scope: string, limit: Record<string, number>, reason: string) => post<Record<string, unknown>>("/actions/budget", { scope, limit, reason }),
  cancel: (task: string, reason: string) => post<Record<string, unknown>>("/actions/cancel", { task, reason }),
  retry: (task: string, note: string) => post<Record<string, unknown>>("/actions/retry", { task, note }),
  priority: (task: string, priority: number, reason: string) => post<Record<string, unknown>>("/actions/priority", { task, priority, reason }),
  createTask: (type: string, goal: string, inputs: string[], priority: number, reason: string) =>
    post<Record<string, unknown>>("/actions/create-task", { type, goal, inputs, priority, reason }),
  pause: (scope: string, reason: string) => post<Record<string, unknown>>("/actions/pause", { scope, reason }),
  resume: (scope: string, reason: string) => post<Record<string, unknown>>("/actions/resume", { scope, reason }),
};

/** Wall-clock "now" for elapsed times. */
export const nowMs = (): number => Date.now();
