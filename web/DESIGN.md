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
| Intervene | Inbox | (count badge only) | WP9b |
| Intervene | Task detail, Pool, Model, Experiments, Improvement | — | WP9b |

Cross-links: a loop instance links to its Case (solve) or Reference (expand); a task links to its
trace in Pipeline; a Solution links to its Result page; a Result links to its Family; a Family
chain node links to its Template record.

## 3. Pages

### 3.1 Overview — "What does the system do, and what is it doing now?"

- **Primary visual:** the three orchestrator state machines drawn from
  `src/kta/orchestrator/machines.yaml` (D96): solve loop, expand loop, task lifecycle. Laid out
  automatically (elkjs layered, left → right) and rendered with React Flow, read-only.
  - Stacking order: solve loop (full width, it is the main loop and needs 9 layers to stay
    legible) → Now band → expand loop → task lifecycle → pipeline strip.
  - Each state node shows: name, what it does (two lines, full text on hover), who works in it
    (orchestrator / executor jobs / agent task / you), and a live **count badge** of instances in
    that state. Nodes with instances are tinted by status tone with a left accent.
  - Transitions that create a task carry a violet pill label (`① T6`, `② T4`, `④ T2/T1`, `T1`);
    other guarded exits carry the short `label` from the YAML. Labels sit at the edge tail, next
    to the state they leave.
  - Under each diagram, a **branches** list spells out every guarded exit in rule order (full
    `when` condition, created task, target), and the YAML `notes` (paths the design leaves open).
  - Dashed edges: resume after waiting, retry, human actions, "any state → cancelled".
  - **Click a state** → an instance list opens under the diagram (subject, round, time in
    state, waiting-on task; task rows show goal, slot, retries, last rejection, card record).
- **Now band** (four columns): agent slots (task, elapsed vs budget, uncharged queue time), GPU
  exclusive lease (holder, held vs 10-min max) and job pools (running/capacity, queued), workers
  and today's budgets, inbox count and recently finished items.
- **Bottom:** compact pipeline strip — the last 45 minutes of spans per lane, no labels, click →
  Pipeline page.
- Collapsed by default: the full transition table under each diagram ("Show transitions as a
  table").

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

- Phase A: `web/src/data/source.ts` defines one async function per API endpoint
  (spec/modules/api-web.md). The only implementation is `sample.ts` over
  `web/src/sample/` (`kta:fixture`, D95). State machines are read from
  `src/kta/orchestrator/machines.yaml` at build time.
- Phase B: the same functions call the real API (`/runs/active`, `/trace`, `/cases/{id}/timeline`,
  `/lineage/{id}`, `/families/{id}/chain`, `/coverage`, `/records/{id}`, `/machines`, SSE
  `/events`) and the sample path is deleted.

## 6. Open questions for the user

- Is the Overview's split (solve loop large, expand + task lifecycle small) the right emphasis?
- Should Pipeline be a top-level page (as now) or a tab inside Case / Running?
- Which pages do you want first in WP9b (Inbox and Task detail are planned)?
