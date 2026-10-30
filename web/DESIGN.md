# KTA web — UI design (WP9a, first version)

Status: assistant's design (D93 asks for a readable, non-minimal frontend; the page split and
layout below are the assistant's default and need the user's review). Source requirements:
`spec/modules/api-web.md`, `docs/design/30-frontend-and-supervision.md`, D67, D93–D96.

The frontend has three jobs (design/30): **watch** what the system is doing, **understand** why a
result holds, **intervene** where you are needed. WP9a is read-only; intervene controls come in
WP9b and are shown as disabled placeholders only where their absence would confuse.

## 1. Principles

1. **One question, one primary visual per page.** Each page states its question in the header
   and gives one visual most of the space. Everything else is a secondary panel, smaller and
   lower in contrast.
2. **Every number is a link.** Any figure or conclusion is rendered with `<RecordLink>`; clicking
   opens the record drawer (`?rec=<id>`, deep-linkable) with the envelope (kind, origin, actor,
   created_at, status, assurance), body, refs (each a link again) and blobs.
3. **Details on demand.** Long lists, raw bodies, and internal agent events are collapsed by
   default; the default view answers the page question without scrolling on a 1440×900 screen.
4. **Status colour means the same thing everywhere** (§4.2). Status is never colour alone: it
   always carries a text label or icon.
5. **Honest data provenance.** While the store does not exist, every page shows the SAMPLE DATA
   banner (D95). Records with `origin ≠ real` carry an origin chip wherever they appear.
6. **Desktop only, single user, light theme** (design/30 §6). Minimum width 1280 px.

## 2. Information architecture

Left sidebar, grouped by the three jobs. Top bar: page title + question, pool snapshot
selector (applies to every chart that depends on the pool), data-as-of time, live indicator.

| Group | Page | Route | Version |
| --- | --- | --- | --- |
| Watch | Overview (home) | `/` | WP9a |
| Watch | Running | `/running` | WP9a |
| Watch | Pipeline | `/pipeline` | WP9a |
| Understand | Cases → Case | `/cases`, `/cases/:id` | WP9a |
| Understand | Result | `/results/:id` | WP9a |
| Understand | Families → Family | `/families`, `/families/:id` | WP9a |
| Understand | Coverage | `/coverage` | WP9a |
| Understand | Pool, Model | `/pool`, `/model` | WP9b |
| Intervene | Inbox, Operations, Experiments, Improvement, Agents | `/inbox`, `/operations`, `/experiments`, `/improvement`, `/agents` | WP9b |
| (linked) | Task detail | `/tasks/:id` | WP9b |

Cross-links: a loop instance links to its Case (solve) or Reference (expand); a task links to its
trace in Pipeline; a Solution links to its Result page; a Result links to its Family; a Family
chain node links to its Template record.

## 3. Pages

### 3.1 Overview — "What does the system do, and what is it doing now?"

Revised after the coordinator's review (D174): the machine and the Now column are both above the
fold at 1440×900.

- **Primary visual:** one orchestrator state machine at a time, chosen with tabs (Solve loop by
  default, Expand loop, Task lifecycle; each tab shows its instance count; `?machine=` keeps the
  choice in the URL). Drawn from `src/kta/orchestrator/machines.yaml` (D96) with elkjs layered
  layout wrapped into two rows and React Flow; never shrunk below 80 % (wider graphs scroll).
  - Every node shows its name, its full "does" text (up to four lines), who works in it and a
    live count badge; nodes with instances are tinted by status tone.
  - Task-creating transitions carry violet labels (`① T6`, `② T4`, `④ T2/T1`, `T1`); other
    guarded exits carry the YAML `label`, shortened to the rule glyph when the edge is too short.
  - "From any state" transitions (you cancel) are a footnote line under the diagram, not a
    floating node.
  - Click a state → its instances are listed under the diagram. Below: the branch conditions in
    rule order, the YAML notes, and the full transition table (collapsed).
- **Now column (right, 320 px):** needs-you items (from `/inbox`), agent slots with elapsed vs
  budget, GPU exclusive lease, workers with pool occupancy pills, today's budgets, just finished.
- **Bottom:** recent pipeline strip (45 minutes, 17 px per track, click → Pipeline).

### 3.2 Running — "What is occupying the system right now?"

- **Primary visual:** resource board. One row per resource (agent slots, gpu-exclusive lease,
  gpu-shared, cpu pool per worker), each slot a card showing occupant, kind colour, elapsed time,
  budget bar. Empty slots are drawn as dashed placeholders so capacity is visible.
- Secondary: active loop instances table (kind, subject, state chip, round k/5, time in state,
  waiting on); task queue (queued tasks with priority and retries); worker table (host,
  transport, labels, commit, last seen, health); budget tree (global → process → loop → task)
  as horizontal used/reserved/remaining bars.
- Collapsed: finished-in-last-hour list.

### 3.3 Pipeline — "In what order and in parallel did things run, and where did time go?"

- **Primary visual:** swimlane Gantt (ECharts custom series). X = time, Y = lanes
  (`process:*`, `agent:*`, `worker:*`, `gpu:*`, `human`). One bar per span coloured by span kind
  (§4.3); queue/lock-wait spans are drawn light, work spans solid; failed spans get a red outline.
  Zoom (wheel, slider) and pan; "Follow now" toggle.
- Top header: utilisation per lane in the window, GPU idle %, agent think-vs-tool share.
- Clicking an `agent_session` span expands its inner `agent_model` / `tool_call` segments
  in-place (sub-row under the lane). Hover tooltip: name, start/end, duration, status, key attrs.
  Clicking any span opens the span detail panel with links to its task/job/record.
- Filters in one row above the chart: trace (case / reference / experiment run), lane group,
  span kinds.
- Collapsed: span table view.

### 3.4 Cases — list; Case — "How well is this problem solved?"

- **List:** one row per case with state, round dots (judged / ended waiting on a task / current /
  left before escalation), a sparkline of % of bound per round, bound, accepted result, the
  active task and the last activity.

- **Header stat row:** lower bound, strongest reference, best result, % of bound — each a
  RecordLink.
- **Primary visual:** iteration timeline. One row per round; each row is the sequence of
  solve-loop states as segments over time, with task markers (T1/T4/T6…) and rejection markers
  (× with reason on hover); the judging rule fired at the end of each round is labelled.
- Secondary: per-shape-bucket results (bar: best vs references vs bound, in µs), candidate
  shortlist of the last round (predicted vs measured), Gaps raised by this case.
- Collapsed: full task list with retries.

### 3.5 Result — "Why is this result trustworthy?"

- **Primary visual:** evidence chain, top to bottom: Template versions (via Family chain) →
  assignment → prediction → build + SASS check → correctness check → measurement → gap to bound.
  Each link is a card with its record ID, status/assurance chip, and the one fact that matters
  (e.g. "regs 96, no spills", "passed 6/6 distributions", "median 41.3 µs ±0.4").
- Beside it: gap decomposition — per hardware unit bound vs predicted vs measured (bars), with
  the explained / unexplained share.
- Collapsed: raw bodies, build log summary, failures list.

### 3.6 Families — list; Family — "How is this kernel built, step by step?"

- **List:** one row per family with a mini chart of its chain (one bar per step spanning the
  refinement levels it covers, coloured by assurance), versions, solutions, last version time.

- **Primary visual:** the composition chain as a DAG (React Flow + elkjs, layered left→right):
  node = chain item (template@version, the transaction it refines, assurance chip), edge =
  `after`. Level band colouring on nodes (refinement level 0–5).
- Click a node → Template card panel: summary, level_from→level_to, change kind, variables,
  constraints, obligations, requires_hw, expected SASS.
- Secondary: family variables and glue constraints, version list with diff summary, solutions
  using this family.

### 3.7 Coverage — "Have we discovered enough, and where are the blind spots?"

- **Always at top:** sampling scope ("Sources: … , N in total", by category) — required by spec.
- **Primary visual:** discovery curve — sources decomposed vs cumulative distinct Templates
  (mean + band over reshuffles), Chao1 estimate line with its interval.
- Secondary grid: coverage-rate trend; instruction-family coverage table (red rows = SOTA uses,
  we cannot produce); hardware feature × pattern grid; reference coverage matrix (reference ×
  expand-loop state); gap explanation per case; learning backlog.
- Snapshot selector switches all of them.

## 4. Visual system

Tokens live in `web/src/styles/tokens.css`. Palette and status colours follow the validated
default data-viz palette (categorical order fixed; status colours reserved).

### 4.1 Typography and spacing

- Font: `system-ui, -apple-system, "Segoe UI", sans-serif`; IDs and code in
  `ui-monospace, SFMono-Regular, Menlo, monospace`.
- Scale (px): 11 caption · 12 small/labels · 13 body · 15 section title · 18 page title ·
  28 hero number. Weights 400/500/600. Tabular figures in tables and axes.
- Spacing scale 4 · 8 · 12 · 16 · 24 · 32. Panels: 12 px padding, 8 px radius, hairline border.
- Ink: primary `#0b0b0b`, secondary `#52514e`, muted `#898781`. Surfaces: page `#f6f6f3`,
  panel `#fcfcfb`, hairline `#e1e0d9`.

### 4.2 Status colours (one meaning everywhere)

| Token | Meaning | Colour | Applies to |
| --- | --- | --- | --- |
| `--st-running` | in progress, active | blue `#2a78d6` | loop state with instances, task `running`/`leased`, span `running` |
| `--st-waiting` | waiting, queued | amber `#fab219` | `waiting`, `queued`, lock wait |
| `--st-ok` | accepted, passed, healthy | green `#0ca30c` | `accepted`, Check passed, worker healthy |
| `--st-fail` | failed, rejected | red `#d03b3b` | `rejected`, Check failed, span `fail` |
| `--st-human` | needs you | violet `#6250d6` | `escalated`, inbox, human lane |
| `--st-idle` | cancelled, idle, unknown | gray `#898781` | `cancelled`, empty slot |

Status chips always show the text label next to a coloured dot.

### 4.3 Span kind colours (Pipeline, strips, resource board)

Grouped by what does the work so that a lane reads at a glance:

| Group | Kinds | Colour |
| --- | --- | --- |
| Orchestration | `loop_step` | slate `#6b7280` |
| Agent | `task`, `agent_session` | violet `#6250d6`; `agent_model` light violet; `tool_call` aqua `#1baf7a` |
| Build | `compile` | orange `#eb6834` |
| Verify | `check` | green `#008300` |
| Measure | `measure`, `profile` | blue `#2a78d6` |
| Solve | `solve` | magenta `#e87ba4` |
| Waiting | `job_queue`, `lock_wait`, `human_wait` | the group colour at 25 % opacity, hatched |
| Holding | `job_run`, `lock_hold` | yellow `#eda100` |

Unknown kinds (the table is open) get gray and are listed as "other" in the legend.

### 4.4 Diagram conventions (state machines and chains)

- Node kinds: deterministic step (rectangle), wait state (rectangle, dashed border), terminal
  (rounded, double border), task pill (small, violet outline).
- Edges: solid = normal transition; violet = creates a task; dashed = resume after waiting.
- Rule numbers ①–④ show evaluation order on judging's outgoing edges.

## 5. Data access

- `web/src/data/source.ts` has one async function per API endpoint; the Vite dev server proxies
  `/api` to the KTA API (`scripts/api`, D173). Pages never hold data of their own.
- Live updates: `web/src/data/live.tsx` listens to SSE `/api/events`; every event bumps a data
  version (throttled to 2 s) and all `useApi` calls re-fetch, keeping the old data on screen.
- Provenance: `GET /meta` returns record counts by origin; any fixture/simulated records turn on
  the FIXTURE DATA banner (D168). Missing evidence links are listed on the Result page.
- Phase A's sample dataset (`web/src/sample/`) and build-time YAML plugin are deleted.

## 6. Second version (WP9b): intervene pages and operations

Status: assistant's design (D348, needs the user's review). Every control posts to
`/actions/*` and every call writes a HumanAction record (D342).

### 6.1 Controls

- **One dialog for every operation** (`web/src/components/actions.tsx`): title naming the
  subject, what will happen, the records it writes (from `GET /actions/capabilities`), a reason
  field (marked *required* for reject, deprecate, budget and retry; the API answers 400 without
  one), extra fields when needed (budget limit). The result opens in a separate modal held at
  the top level (D359), because its subject may leave the page when the data refreshes: the
  HumanAction ID as a record link, the new task state with a link to the task, the records
  marked stale, follow-ups that could not be created, a batch check's problems.
- **Unwired operations are visible, not hidden or faked** (D341): drawn as dashed, disabled
  buttons tagged "not wired" whose tooltip names what they need (after D355: priority, pause,
  resume). A 501 that still reaches the dialog says "Not available yet. Nothing was written."
  What a wired operation does not do yet (`partial`) is shown in its dialog and on the Inbox.
- **Controls sit next to their subject**: the record drawer offers flag / reject / deprecate /
  confirm / approve for any record; Running and Task offer cancel / retry; the Result page flag /
  reject; Experiments approve-and-freeze; Improvement approve; Agents "make current".
- Danger actions (reject, cancel) are outlined red; approvals are violet (the "needs you" colour).

### 6.2 Pages

| Page | Question | Primary visual | Secondary |
| --- | --- | --- | --- |
| Inbox `/inbox` | What needs me now? | the item list, most urgent first; each item: trigger, subject, waiting time, last rejection, the system's suggestion, its controls | the ten design/30 §2 triggers and which ones are derived today |
| Task `/tasks/:id` | What did this agent do? | the task's own swimlane (session, model and tool segments, jobs) | time stats, kta calls table, task card, done checks with failures, AgentRuns with the prompt (TASK.md) blob, records written, your actions |
| Operations `/operations` | What can I start, stop or change? | start-a-task form (task type cards T1–T6, goal, input records with family quick-picks, priority) | pause / resume by scope; budget editor per scope and resource; action log; capability table |
| Pool `/pool` | What does the system know now? | tabbed tables: Templates, Patterns (`other` rows highlighted), Structures, Gaps, References (reconstruction bar) with per-row actions | count tiles; snapshot filter (default: members of the selected snapshot) |
| Model `/model` | Can the model be trusted? | predicted vs measured scatter with ±20 % guides, per model version | error stats; largest errors with record links |
| Experiments `/experiments` | How far are the experiments? | one card per protocol: arms with run progress, approval state, results table | runs list, protocol parameters |
| Improvement `/improvement` | Is the system getting better? | stacked bars of Gap attribution per day | attribution legend with meanings; improvement reports with items and approval |
| Agents `/agents` | What controls the worker agents? | runtime skills × task types matrix (current profile) and skill cards | profile versions as a timeline with changed fields; pointer history; batch environments with their lock and verify state |

## 7. Open questions for the user

- Is the Overview's split (solve loop large, expand + task lifecycle small) the right emphasis?
- Should Pipeline be a top-level page (as now) or a tab inside Case / Running?
- Should operations ask for a reason always (the spec requires it only for reject, deprecate and
  budget), and should the operator name come from config (D342)?
