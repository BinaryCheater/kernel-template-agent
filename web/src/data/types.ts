// Shapes returned by the KTA API (spec/modules/api-web.md). Phase A fills them from the sample
// dataset; phase B from the real endpoints named on each type.

export type Origin = "real" | "fixture" | "simulated" | "historical";

/** GET /meta */
export interface Meta {
  env: string;
  now: string;
  origins: Partial<Record<Origin, number>>;
}

/** GET /records/{id} — record envelope (spec/schemas.md §1) plus derived status. */
export interface RecordEnvelope {
  id: string;
  kind: string;
  schema: number;
  created_at: string;
  actor: { type: string; name: string; task?: string | null; commit?: string | null; dirty?: boolean | null };
  refs: Record<string, string | string[]>;
  body: Record<string, unknown>;
  blobs: Record<string, string>;
  origin: Origin;
  status?: string;
  assurance?: string | null;
}

// ---- state machines (GET /machines, from src/kta/orchestrator/machines.yaml) ----

export type StateKind = "step" | "wait" | "terminal";
export type Work = "orchestrator" | "executor" | "agent" | "human";
export type TransitionKind = "normal" | "creates_task" | "resume" | "retry" | "human" | "any";

export interface StateDef {
  id: string;
  kind: StateKind;
  work: Work;
  does: string;
  initial?: boolean;
}
export interface TransitionDef {
  from: string;
  to: string;
  kind: TransitionKind;
  when?: string;
  label?: string;
  rule?: number;
  creates?: string[];
}
export interface MachineDef {
  id: "solve" | "expand" | "task";
  title: string;
  instance_of: string;
  process: string;
  summary: string;
  limits?: Record<string, number | string>;
  states: StateDef[];
  transitions: TransitionDef[];
  notes?: string[];
}
export interface TaskTypeDef {
  id: string;
  name: string;
  by: string;
  delivers: string;
}
export interface Machines {
  version: number;
  task_types: TaskTypeDef[];
  machines: MachineDef[];
}

// ---- GET /runs/active ----

export interface LoopInstance {
  id: string;
  machine: "solve" | "expand";
  subject: string; // Case or Reference record id
  subject_name: string;
  state: string;
  round?: number | null;
  max_rounds?: number | null;
  started_at: string;
  state_since: string;
  waiting_on?: string | null; // task id
  escalated?: boolean;
  trace_id: string;
}

export type TaskType = "T1" | "T2" | "T3" | "T4" | "T5" | "T6";
export interface TaskInstance {
  id: string;
  card: string; // TaskCard record id
  type: TaskType;
  goal: string;
  state: string;
  slot?: string | null; // "agent:1"
  ops_state?: string;
  loop_id: string;
  created_at: string;
  state_since: string;
  retries: number;
  max_retries: number;
  budget_wall_s: number;
  used_wall_s: number;
  queue_s: number;
  priority: number;
  last_rejection?: string | null;
}

export interface Job {
  id: string;
  kind: string; // compile | check | measure | profile | run
  name: string;
  pool: "cpu" | "gpu-shared" | "gpu-exclusive";
  worker: string;
  state: "queued" | "running";
  queued_at: string;
  started_at?: string | null;
  task_id?: string | null;
  loop_id?: string | null;
}

export interface Lease {
  gpu: string; // worker:device
  holder_job: string;
  since: string | null;
  max_s: number;
  until?: string;
}

export interface WorkerInfo {
  name: string;
  host: string;
  transport: string | null;
  labels: Record<string, string>;
  commit: string | null;
  last_seen: string;
  health: "ok" | "degraded" | "down";
  record: string | null;
  pools: { pool: string; capacity: number; running: number; queued: number }[];
}

export interface BudgetNode {
  key: string; // global | process:<name> | loop:<id> | task:<id>
  label: string;
  unit: string;
  limit: number;
  used: number;
  reserved: number;
  children?: BudgetNode[];
}

export interface ActiveRuns {
  as_of: string;
  agent_slots: string[];
  loops: LoopInstance[];
  tasks: TaskInstance[];
  jobs: Job[];
  leases: Lease[];
  workers: WorkerInfo[];
  budgets: BudgetNode[];
  inbox_count: number;
  recently_finished: { id: string; label: string; status: string; ended_at: string; link: string }[];
}

// ---- GET /trace ----

export interface Span {
  span_id: string;
  parent_id: string | null;
  trace_id: string;
  kind: string;
  lane: string;
  name: string;
  start_ts: string;
  end_ts: string | null;
  status: "ok" | "fail" | "cancelled" | "running";
  attrs: Record<string, string | number>;
}

export interface TraceInfo {
  id: string;
  label: string;
  kind: string;
}

// ---- GET /cases, /cases/{id}/timeline ----

export interface CaseSummary {
  id: string;
  name: string;
  problem: string;
  loop_id: string | null;
  state: string;
  round: number;
  best_us?: number | null;
  bound_us: number | null;
  bound_record: string | null;
  pct_of_bound?: number | null;
  objective: string;
  /** GET /cases only: per-round progress, the active task and the last activity. */
  rounds?: unknown;
  active_task?: { type: string; state: string } | null;
  last_activity?: string | null;
}

export interface CaseRoundProgress {
  n: number;
  last_state: string;
  open: boolean;
  rule: number | null;
  tasks: string[];
  pct_of_bound: number | null;
}
export interface CaseIndexRow extends CaseSummary {
  rounds: CaseRoundProgress[];
}

export interface RoundStep {
  state: string;
  start: string;
  end: string | null;
  note?: string | null;
}
export interface RoundTask {
  id: string;
  type: TaskType;
  card: string;
  created_at: string;
  ended_at: string | null;
  state: string;
  rejections: { at: string; reason: string; record: string }[];
}
export interface Round {
  n: number;
  steps: RoundStep[];
  tasks: RoundTask[];
  outcome?: { rule?: number | null; label: string; record: string } | null;
}
export interface BucketResult {
  bucket: string;
  bound_us: number;
  best_us?: number | null;
  references: Record<string, number>;
  solution?: string | null;
}
export interface Candidate {
  rank: number;
  assignment: Record<string, number | string>;
  predicted_us: number | null;
  measured_us?: number | null;
  build_ok?: boolean | null;
  check_passed?: boolean | null;
  record: string;
}
export interface CaseDetail extends CaseSummary {
  record: string;
  shapes: string;
  references: { name: string; us: number; record: string }[];
  best?: { us: number | null; record: string | null; solution: string; family: string | null } | null;
  rounds: Round[];
  buckets: BucketResult[];
  candidates: Candidate[];
  gaps: { id: string; description: string }[];
}

// ---- GET /lineage/{id} ----

export interface EvidenceLink {
  stage: string;
  title: string;
  fact: string;
  record: string;
  status?: string | null;
  origin?: string;
}
export interface UnitGap {
  unit: string;
  bound_us: number;
  predicted_us: number;
  measured_us: number;
}
export interface ResultDetail {
  id: string; // Solution record
  case_id: string;
  case_name: string;
  family: string | null;
  family_name: string;
  bucket: string;
  measured_us: number | null;
  bound_us: number | null;
  pct_of_bound: number;
  chain: EvidenceLink[];
  units: UnitGap[];
  explained_share: number | null;
  verdict: string | null;
  missing: string[];
  judgement: string | null;
}

// ---- GET /families/{id}/chain ----

export interface ChainNode {
  key: string;
  template: string; // name@version
  template_record: string | null;
  applies_to: string;
  after: string[];
  level_from: number | null;
  level_to: number | null;
  change: string | null;
  assurance: string;
  summary: string;
  requires_hw: string[];
  variables: { name: string; domain: string; binding: string }[];
  bindings: Record<string, string | number | boolean>;
}
export interface FamilySummary {
  id: string;
  name: string;
  version: number;
  backend: string;
  problem: string;
  steps: number;
  solutions: number;
  step_detail?: { key: string; template: string; level_from: number | null; level_to: number | null; assurance: string }[];
  versions?: number | { version: number; record: string; note: string; created_at: string }[];
  updated_at?: string;
}
export interface FamilyDetail extends FamilySummary {
  workload: string;
  chain_record: string | null;
  nodes: ChainNode[];
  variables: { name: string; domain: string; meaning: string }[];
  glue: { id: string; expr: string; meaning: string }[];
  versions: { version: number; record: string; note: string; created_at: string }[];
  solution_ids: { id: string; label: string }[];
}

// ---- GET /coverage?snapshot= ----

export interface Snapshot {
  id: string;
  label: string;
  created_at: string;
  members: number;
  current: boolean;
}
export interface CoverageReport {
  snapshot: string;
  record: string;
  scope: { description: string; total: number; by_category: Record<string, number> };
  discovery: {
    sources: number[];
    mean: number[];
    lo: number[];
    hi: number[];
    chao1: { estimate: number; lo: number; hi: number };
  };
  trend: { snapshot: string; good_turing: number; expressible: number }[];
  instruction_families: { family: string; we_produce: boolean; sota_uses: boolean; note: string }[];
  hw_patterns: { features: string[]; patterns: string[]; cells: { f: number; p: number; n: number; best: string }[] };
  references: { name: string; state: string; record?: string; gaps: number }[];
  gap_explained: { case: string; explained: number; unexplained: number }[];
  backlog: { item: string; why: string; priority: number }[];
  structure: { other_patterns: number; unmatched_by_kind: Record<string, number>; assurance: Record<string, number> };
}

export interface InboxItem {
  id: string;
  trigger: string;
  subject: string;
  suggestion: string;
  record: string;
  at: string;
}
