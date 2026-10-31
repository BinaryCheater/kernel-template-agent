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
  outcome?: { rule?: number | null; label: string; record: string; gap_class?: string | null; decomposition?: string | null } | null;
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
  requires: string[]; // abstract Capabilities (D442)
  probed_on: string[];
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

// ---- GET /pool/diff?a=&b= (D449) ----

export interface PoolDiffEntry { id: string; kind: string; label: string }
export interface PoolDiff {
  a: string; b: string; a_at: string; b_at: string;
  counts_a: Record<string, number>; counts_b: Record<string, number>;
  added: PoolDiffEntry[]; removed: PoolDiffEntry[];
  changed: { kind: string; label: string; old: string; new: string }[];
  status: { id: string; kind: string; label: string; old: string; new: string }[];
}

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
  /** Quantities that need real runs, by name -> what they need (WP10b D386). */
  pending?: Record<string, string>;
  /** The full CoverageReport of kta.knowledge.coverage (WP10b D387). */
  report?: {
    direct_test?: { references: number; all_steps_matched: number; nothing_unexplained: number };
    gaps_by_missing?: Record<string, number>;
    gaps_by_attribution?: Record<string, number>;
    templates_missing?: unknown[];
  };
  /** Learning-loop coverage (WP20 D567): SOTA levels 1-4, system features counted apart,
   *  language gaps counted at admission, per-backend expressibility and intent fidelity. */
  learning?: {
    sota_levels: { level: number; name: string; meaning: string; measured_by: string; references: number; of: number; pending: string | null; step_rate?: number | null }[];
    system_features: { total: number; by_feature: Record<string, number> };
    language_gaps: { total: number; by_kind: Record<string, number>; by_reference: Record<string, number>; admitted_references: number };
    backends: Record<string, { steps: number; expressible: number; rate: number | null; intent_declared: number; intent_confirmed: number | null; note: string }>;
  };
}

export interface InboxItem {
  id: string;
  trigger: string;
  subject: string;
  suggestion: string;
  record: string | null;
  at: string | null; // when the task entered its state; null for items not about a task
  task?: string | null; // ops task id when the item is about a task
  detail?: string | null;
  actions?: string[]; // operation names from GET /actions/capabilities
}

// ---- WP9b: operations (POST /actions/*, GET /actions/capabilities, GET /actions/log) ----

export interface Capability {
  name: string; // reject, flag, cancel, ...
  endpoint: string;
  writes: string;
  wired: boolean;
  needs: string | null; // what is missing when not wired
  partial?: string;
}

export interface ActionLogEntry {
  id: string;
  at: string;
  by: string;
  actor_type: string;
  task: string | null;
  action: string;
  verb: string;
  params: Record<string, unknown> | null;
  args: string;
  reason: string;
  refs: Record<string, string | string[]>;
  origin: Origin;
}

// ---- GET /tasks/{id} ----

export interface TaskSpan {
  span_id: string;
  parent_id: string | null;
  trace_id: string;
  kind: string;
  lane: string;
  name: string;
  start: string;
  end: string | null;
  dur_s: number | null;
  status: string;
  attrs: Record<string, unknown>;
}

export interface TaskRecord {
  id: string;
  kind: string;
  created_at: string;
  actor: string;
  status: string;
  origin: Origin;
  body: Record<string, unknown>;
  blobs: Record<string, string>;
  refs: Record<string, string | string[]>;
}

/** A Report titled done_checks: the orchestrator's verdict on one submission (D276). */
export interface DoneCheckReport {
  id: string;
  created_at: string;
  actor: string;
  passed: boolean;
  summary: string;
  checks: { name: string; passed: boolean; failures: string[]; data: Record<string, unknown> }[];
  reason: string;
}

export interface TaskDetail {
  task: TaskInstance;
  card: { id: string; type: TaskType; goal: string; inputs: string[]; done_checks: string[]; budget: Record<string, number>; profile: string; work_unit: string };
  loop: LoopInstance | null;
  spans: TaskSpan[];
  calls: TaskSpan[];
  records: TaskRecord[];
  agent_runs: TaskRecord[];
  submissions: TaskRecord[];
  checks: DoneCheckReport[];
  rejections: DoneCheckReport[];
  time: { wall_s: number; tool_s: number; queue_s: number; source: "AgentRun" | "spans" };
  human_actions: { id: string; at: string; by: string; action: string; reason: string }[];
}

export interface BlobText {
  sha256: string;
  media: string;
  size: number;
  truncated: boolean;
  text: string;
}

// ---- GET /agents ----

export interface RuntimeSkill {
  name: string;
  description: string;
  lines: number;
  sha256: string;
  files: string[];
  catalogue: Record<string, string> | null;
  used_by: string[];
}

export interface ProfileRow {
  id: string;
  created_at: string;
  origin: Origin;
  status: string;
  current: boolean;
  name: string;
  version: number;
  provider: string;
  cli_version: string;
  auth: string;
  isolation: string;
  model: string;
  reasoning_effort: string | null;
  cli_flags: string[];
  skills: Record<string, string[]>;
  skill_bundles: Record<string, string>;
  allowed_commands: Record<string, number>;
  context_policy: Record<string, unknown>;
  loop_policy: Record<string, unknown>;
  templates: Record<string, string | null>;
  changed: string[] | null;
}

export interface BatchEnv {
  batch: string;
  path: string;
  lock: Record<string, unknown> | null;
  files?: number | null;
  lock_sha256?: string;
  work_units: string[];
  verify: { state: string; needs?: string };
}

export interface AgentsView {
  skills: { has_catalogue: boolean; skills: RuntimeSkill[]; planned: Record<string, string>[]; absent: { name: string; used_by: string[] }[] };
  current: string | null;
  profiles: ProfileRow[];
  history: { at: string; from: string | null; to: string | null }[];
  approvals: { id: string; at: string; by: string; reason: string; refs: Record<string, string | string[]> }[];
  batches: BatchEnv[];
  batches_root: string | null;
}

// ---- GET /pool ----

interface PoolEnv {
  id: string;
  created_at: string;
  origin: Origin;
  actor: string;
  status: string;
  assurance: string | null;
}
export interface PoolPattern extends PoolEnv { name: string; level: number; category: string; description: string; examples: number; confirmed_by: string | null }
export interface PoolTemplate extends PoolEnv {
  name: string; version: number; level_from: number; level_to: number; change: string; refines: string; produces: string[];
  summary: string; requires: string[]; probed_on: string[]; variables: number; constraints: number; obligations: number;
}
export interface PoolStructure extends PoolEnv { name: string; effect: string; description: string; evidence_needed: string; confirmed_by: string | null }
export interface PoolGap extends PoolEnv { missing: string; description: string; level: number | null; pattern_guess: string | null; attribution: string; refs: Record<string, string | string[]> }
export interface PoolReference extends PoolEnv { name: string; runnable_on: string[]; reconstruction: string | null; matched: number | null; total: number | null; unmatched: number | null }
export interface PoolView {
  snapshot: string | null;
  patterns: PoolPattern[];
  templates: PoolTemplate[];
  structures: PoolStructure[];
  gaps: PoolGap[];
  references: PoolReference[];
  counts: Record<string, number>;
  other_patterns: number;
}

// ---- GET /experiments, /improvements ----

export interface ExperimentProtocol extends PoolEnv {
  name: string;
  version: number;
  description: string;
  params: Record<string, unknown>;
  approved: { id: string; at: string; by: string; reason: string }[];
  /** The experiment process's state (WP11 D403), null fields before it starts. */
  progress: { status: string | null; started: number; finished: number; report: string | null; error: string | null };
  arms: { name: string; runs: number; planned: number | null }[];
  runs: (PoolEnv & { name: string; condition: string; seed: number | null; params: Record<string, unknown>; note: string })[];
  results: (PoolEnv & { summary: string; data: Record<string, unknown> })[];
}

export interface ImprovementItem {
  key?: string;
  title: string;
  kind: string;
  description?: string;
}

export interface ImprovementsView {
  attribution: Record<string, number>;
  missing: Record<string, number>;
  attribution_by_day: ({ day: string } & Record<string, number | string>)[];
  reports: (PoolEnv & { summary: string; batch: string | null; attribution: Record<string, number>; items: ImprovementItem[]; approved: { id: string; at: string; by: string; reason: string }[];
    rejected: { id: string; at: string; by: string; reason: string }[]; system_items: number })[];
}

// ---- GET /model/residuals ----

export interface Residual {
  prediction: string;
  measurement: string;
  model: string;
  predicted_us: number;
  measured_us: number;
  rel_error: number;
}

/** GET /quotas: process shares of the global budget (WP11 D401). */
export interface Quotas {
  configured: Record<string, number>;
  in_force: Record<string, number>;
  experiments: string[];
}

// ---- WP21: controls registry and staged release ----

/** One knob of the controls registry (docs/design/12) with the value in force. */
export interface Knob {
  id: string;
  row: string;
  section: string;
  controls: string;
  type: "int" | "float" | "bool" | "str" | "enum" | "per_task_type" | "mapping" | "list_str";
  value: unknown;
  initial: unknown;
  prevents: string;
  where: string[];
  status: "spec" | "default" | "set";
  source: string;
  set_by: string[];
  set_from: string;
  set_by_cell: string;
  editable: boolean;
  accessor: string;
  consumer: string;
  unit: string;
  choices: string[];
  fields: Record<string, string>;
  min: number | null;
  max: number | null;
}

export interface SystemExperiment {
  id: string;
  question: string;
  design: string;
  arms: string[];
  metrics: string[];
  sets: string[];
  profile_fields: string[];
  sets_elsewhere: string[];
  depends_on: string[];
  step: number;
  method: "run" | "logs";
  status: string;
}

/** GET /controls */
export interface ControlsView {
  version: { id: string | null; version: number; stored: boolean; definitions_hash: string; stored_under_hash: string };
  knobs: Knob[];
  experiments: SystemExperiment[];
  write: { header: string; token_configured: boolean };
}

/** GET /controls/versions */
export interface ControlsVersionRow {
  id: string;
  version: number;
  at: string;
  author: string;
  author_type: string;
  reason: string;
  changes: { knob: string; old: unknown; new: unknown }[];
  previous: string | null;
}

export interface ReleaseStage {
  id: string;
  name: string;
  runs: string;
  processes: string[];
  task_types: string[];
  agent_calls: "none" | "gated" | "automatic";
  exit: string[];
}

export interface HeldCall {
  task: string;
  card: string;
  type: string;
  work_unit: string;
  attempt: number;
  released_attempt: number | null;
  held: string | null;
  card_body: { type: string; goal: string; inputs: string[]; done_checks: string[]; budget: Record<string, number>; profile: string; work_unit: string };
  pull_log: unknown;
  pull_log_note: string;
  plan: unknown;
  plan_note: string;
}

/** GET /release */
export interface ReleaseView {
  stage: ReleaseStage;
  stages: ReleaseStage[];
  controls: string | null;
  in_flight: string[];
  held: HeldCall[];
}

// ---- GET /model/scoreboard, /model/versions/{id}/timeline (WP16, D511) ----

export interface RegressionRow {
  kernel: string;
  predicted_us: number[];
  measured_us: number[];
  rank_correlation: number | null;
  error_by_resource: Record<string, number>;
  holdout: boolean;
  predictions: string[];
  measurements: string[];
}

export interface ModelVersionRow {
  id: string;
  label: string;
  target: string;
  hw_model: string;
  arch: string | null;
  simulators: Record<string, string>;
  change: string;
  reason: string;
  admitted: boolean | null;
  summary: Record<string, number>;
  results: RegressionRow[];
  origin: string;
  created_at: string;
  previous: string | null;
}

export interface Scoreboard {
  versions: ModelVersionRow[];
  error_keys: string[];
  fixture: boolean;
}

export interface L2Interval {
  resource: string;
  start_us: number;
  end_us: number;
  phase: string;
  utilisation: number;
}

export interface L2Wave {
  index: number;
  kind: string;
  ctas: number;
  start_us: number;
  end_us: number;
}

export interface TimelineConfig {
  index: number;
  prediction: string | null;
  measurement: string | null;
  configuration: Record<string, unknown> | null;
  predicted_us: number;
  measured_us: number;
  waves: L2Wave[];
  intervals: Record<string, L2Interval[]>;
  busy_us: Record<string, number>;
  idle_us: Record<string, Record<string, number>>;
  bottleneck: string | null;
  origin: string;
}

export interface ModelTimeline {
  version: string;
  label: string;
  kernel: string;
  rank_correlation: number | null;
  error_by_resource: Record<string, number>;
  holdout: boolean;
  origin: string;
  configurations: TimelineConfig[];
}

// ---- GET /system, POST /system, /system/proxy/test, /system/workers/{name}/test (WP22) ----

export interface ProxyEndpoint { scheme: string; host: string; port: number; url: string }
export interface ProxyDetected {
  endpoint: ProxyEndpoint;
  source: "scutil" | "env" | "default" | "override";
  detail: string;
  credentials_dropped: boolean;
  scutil: Record<string, string>;
}
export interface PreflightHost { host: string; port: number; ok: boolean; tunnel?: string; tls?: string; ms?: number; error?: string; attempts?: number }
export interface PreflightResult {
  ok: boolean;
  report?: string;
  skipped?: string;
  proxy: ProxyEndpoint;
  port?: { ok: boolean; ms?: number; error?: string };
  hosts: PreflightHost[];
  error?: string;
}
export interface ProxyStatus { ok: boolean; at: string; error?: string; report?: string; task?: string | null; cleared_by?: string }
export interface ProviderRow {
  name: string;
  proxy: "required" | "off";
  check_hosts: string[];
  cli: { command: string | null; version: string | null };
  auth: { kind: "subscription" | "api"; logged_in: boolean | null; where?: string; note?: string };
  status: ProxyStatus | null;
}
export interface WorkerRow {
  name: string;
  transport: string;
  address: string;
  root: string | null;
  pools: Record<string, number>;
  registered: string | null;
  commit: string | null;
  control_commit: string;
  commit_match: boolean;
  gpu: string | null;
  healthy: boolean | null;
  intake_proxy: string | null;
}
export interface SystemConfigBody {
  revision: number;
  updated_at: string | null;
  updated_by: string | null;
  reason: string | null;
  proxy: { override: string | null; credentials_from_secrets: boolean; no_proxy_extra: string[] };
  providers: Record<string, { proxy: "required" | "off"; check_hosts: string[] }>;
  workers: Record<string, { intake_proxy: string | null }>;
}
export interface SystemView {
  config: SystemConfigBody;
  proxy: {
    detected: ProxyDetected;
    override: string | null;
    in_force: ProxyDetected | null;
    error?: string;
    no_proxy: string[];
    credentials_from_secrets: boolean;
  };
  providers: ProviderRow[];
  workers: WorkerRow[];
  agents: { worker?: string; api_url?: string; batch?: string; error?: string };
  control: { host: string; env: string; data_dir: string; config_file: string; commit: string };
  release: { id: string; name: string; page: string };
  audit: { at: string; by: string; reason: string; verb: string; params?: Record<string, unknown> }[];
  write: { header: string; token_configured: boolean };
}
export interface ProxyTestResult { ok: boolean; from: string; proxy: ProxyDetected; results: Record<string, PreflightResult> }
export interface WorkerTestResult {
  ok: boolean; worker: string; ran: boolean; error?: string; job?: string; exit_code?: number;
  error_kind?: string; wall_s?: number; queue_s?: number; stdout?: string;
}

// ---- GET /mechanisms (WP22, D591) ----

export interface MechRow { id: string; at: string; origin: string }
export interface PlanDiffRow { section: string; key: string; default: unknown; plan: unknown; cited: boolean }
export interface MechPlan extends MechRow {
  family: string; author: string; chain_source: string; case: string | null; default: string | null;
  diff: PlanDiffRow[]; uncited: string[];
}
export interface MechWhatIf extends MechRow {
  model: string; summary: string; peaks: Record<string, number>; walls: Record<string, number>;
  bottleneck: string | null; shadow_price_us: number | null; delta_total_us: number | null;
  base_total_us: number | null; moved_total_us: number | null;
}
export interface MechCanary extends MechRow {
  batch: string; kind: string; expect: string; got: string | null; verdict: string; bucket: string | null; escalate: boolean | null;
}
export interface MechSelection extends MechRow {
  case: string | null; round: number | null; bucket: string;
  model_best_us: number | null; model_top_e_us: number | null; random_best_us: number | null; random_n: number;
  copied_us: number | null; copied_from: string | null; model_beaten_by_random: boolean;
}
export interface MechStage { name: string; status: string; why: string; steps: number }
export interface MechReference extends MechRow {
  name: string; target: string; source_class?: string; priority?: number; tiers?: string[]; stages?: MechStage[]; error?: string;
}
export interface MechanismsView {
  plans: MechPlan[];
  what_if: MechWhatIf[];
  canaries: { rows: MechCanary[]; by_kind: Record<string, Record<string, number>> };
  selection: MechSelection[];
  learning: MechReference[];
  stages: string[];
  experiments: { count: number; by_status: Record<string, number>; page: string };
}

// ---- GET /workflow, POST /workflow/preview, /workflow/drafts (WP22 Part D, D593–D597) ----

export interface WfCreator { kind: "loop" | "process"; machine?: string; from?: string; to?: string; label?: string | null; when?: string | null; rule?: number | null; where?: string }
export interface WfTaskType {
  id: string; name: string; by: string; delivers: string;
  creators: WfCreator[]; step_template: string; skills: string[]; commands: string[];
  done_checks: { name: string; source: string }[];
}
export interface WfKnob { id: string; row: string; value: unknown; editable: boolean; type: string; unit?: string | null; prevents: string }
export interface WfTransition { from: string; to: string; kind: string; label?: string; when?: string }
export interface WorkflowView {
  profile: {
    id: string; name: string; version: number; provider: string; model: string; reasoning_effort: string | null;
    cli_version: string; isolation: string; auth: string; current: boolean; base: string | null; origin: string;
  };
  versions: { id: string; version: number; current: boolean; base: string | null; author: string; author_type: string; origin: string }[];
  task_types: WfTaskType[];
  lifecycle: { states: { id: string; kind: string; does: string }[]; transitions: WfTransition[]; limits: Record<string, unknown> };
  session: { rules: { mode: string; reason: string }[]; source: string };
  templates: Record<string, { blob: string; text: string; placeholders: string[] | null }>;
  skills: Record<string, { ref: string; text: string; used_by: string[] }>;
  knobs: { version: string | null; nodes: Record<string, WfKnob[]> };
  proxy: { provider: string; proxy: "required" | "off"; check_hosts: string[] };
  invariants: { what: string; source: string }[];
  write: { header: string };
}
export interface WfChanges { templates: Record<string, string>; skills: Record<string, string>; fields: Record<string, string | null> }
export interface WfPreview { errors: string[]; warnings: string[]; card: string; task_md?: string }
export interface WfDiff { what: string; old?: unknown; new?: unknown; diff?: string }
