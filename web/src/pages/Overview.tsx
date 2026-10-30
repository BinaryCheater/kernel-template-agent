import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router";
import { api } from "../data/source";
import type { ActiveRuns, MachineDef, Machines } from "../data/types";
import { useApi } from "../data/useApi";
import { useNow } from "../data/live";
import { Branches, DiagramLegend, StateMachine } from "../components/StateMachine";
import { Chip, Loading, Panel, RecordLink, UsageBar } from "../components/ui";
import { dur, hhmm, since } from "../components/format";
import { Swimlanes } from "../components/Swimlanes";

export function Overview() {
  const machines = useApi(api.machines, []);
  const active = useApi(api.active, []);
  if (!machines.data || !active.data) return <Loading error={machines.error ?? active.error} />;
  return <OverviewBody m={machines.data} a={active.data} />;
}

function counts(machine: MachineDef, a: ActiveRuns): Record<string, number> {
  const out: Record<string, number> = {};
  const states = machine.id === "task" ? a.tasks.map((t) => t.state) : a.loops.filter((l) => l.machine === machine.id).map((l) => l.state);
  for (const s of states) out[s] = (out[s] ?? 0) + 1;
  return out;
}

function OverviewBody({ m, a }: { m: Machines; a: ActiveRuns }) {
  const [sel, setSel] = useState<{ machine: string; state: string }>();
  const byId = (id: string) => m.machines.find((x) => x.id === id)!;
  const solve = byId("solve");
  const expand = byId("expand");
  const task = byId("task");
  const c = useMemo(() => ({ solve: counts(solve, a), expand: counts(expand, a), task: counts(task, a) }), [a, solve, expand, task]);
  const pick = (machine: string) => (state: string | undefined) => setSel(state ? { machine, state } : undefined);
  const selFor = (machine: string) => (sel?.machine === machine ? sel.state : undefined);

  return (
    <div className="stack">
      <Intro a={a} />
      <MachinePanel def={solve} m={m} counts={c.solve} selected={selFor("solve")} onSelect={pick("solve")} a={a} primary nodeSize={{ w: 120, h: 80 }}
        options={SOLVE_SPACING} />
      <NowPanel a={a} />
      <MachinePanel def={expand} m={m} counts={c.expand} selected={selFor("expand")} onSelect={pick("expand")} a={a} nodeSize={{ w: 120, h: 66 }} />
      <MachinePanel def={task} m={m} counts={c.task} selected={selFor("task")} onSelect={pick("task")} a={a} nodeSize={{ w: 112, h: 66 }} />
      <DiagramLegend />
      <PipelineStrip />
    </div>
  );
}

function Intro({ a }: { a: ActiveRuns }) {
  const running = a.tasks.filter((t) => t.state === "running").length;
  const solving = a.loops.filter((l) => l.machine === "solve" && l.state !== "accepted").length;
  const learning = a.loops.filter((l) => l.machine === "expand" && l.state !== "covered").length;
  return (
    <div className="intro">
      <p>
        KTA improves GPU kernels by reusing <b>Templates</b> — refinement steps learned from reference implementations. Deterministic code
        drives three loops; LLM agents are called only for creative tasks (the violet <span className="tag task">creates T…</span> steps), and
        every agent proposal must pass fixed checks before the loop moves on.
      </p>
      <div className="intro-now">
        <span><b className="num">{solving}</b> cases being solved</span>
        <span><b className="num">{learning}</b> references being learned</span>
        <span><b className="num">{running}</b>/{a.agent_slots.length} agent slots busy</span>
        <span><b className="num">{a.jobs.filter((j) => j.state === "running").length}</b> jobs running, <b className="num">{a.jobs.filter((j) => j.state === "queued").length}</b> queued</span>
        <span className="needs-you"><b className="num">{a.inbox_count}</b> items need you</span>
      </div>
    </div>
  );
}

const SOLVE_SPACING = { "elk.layered.spacing.nodeNodeBetweenLayers": "58" };

function MachinePanel(p: {
  def: MachineDef; m: Machines; counts: Record<string, number>; selected?: string; onSelect: (s?: string) => void; a: ActiveRuns;
  primary?: boolean; nodeSize: { w: number; h: number }; options?: Record<string, string>;
}) {
  const total = Object.values(p.counts).reduce((x, y) => x + y, 0);
  const lim = p.def.limits ? Object.entries(p.def.limits).map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`).join(" · ") : "";
  return (
    <Panel primary={p.primary} title={p.def.title}
      sub={`one per ${p.def.instance_of}${p.def.process !== "any" ? ` · process ${p.def.process}` : ""}${lim ? ` · ${lim}` : ""}`}
      right={<span className="muted">{total} {p.def.id === "task" ? "tasks" : "instances"} · click a state</span>}>
      <p className="machine-summary">{p.def.summary}</p>
      <StateMachine machine={p.def} taskTypes={p.m.task_types} counts={p.counts} selected={p.selected} onSelect={p.onSelect} nodeSize={p.nodeSize} options={p.options} />
      {p.selected && <Instances def={p.def} state={p.selected} a={p.a} close={() => p.onSelect(undefined)} />}
      <Branches machine={p.def} taskTypes={p.m.task_types} />
      {p.def.notes?.map((n) => <p key={n} className="small muted" style={{ margin: "6px 0 0" }}>Open in design: {n}</p>)}
      <details>
        <summary>Show transitions as a table</summary>
        <table className="t">
          <thead><tr><th>from</th><th>to</th><th>kind</th><th>when</th><th>creates</th></tr></thead>
          <tbody>
            {p.def.transitions.map((t, i) => (
              <tr key={i}><td className="mono">{t.from}</td><td className="mono">{t.to}</td><td>{t.rule ? `rule ${t.rule} · ` : ""}{t.kind}</td>
                <td>{t.when ?? ""}</td><td>{t.creates?.join(", ") ?? ""}</td></tr>
            ))}
          </tbody>
        </table>
      </details>
    </Panel>
  );
}

function Instances({ def, state, a, close }: { def: MachineDef; state: string; a: ActiveRuns; close: () => void }) {
  const now = useNow();
  const nav = useNavigate();
  const doc = def.states.find((s) => s.id === state);
  const loops = def.id === "task" ? [] : a.loops.filter((l) => l.machine === def.id && l.state === state);
  const tasks = def.id === "task" ? a.tasks.filter((t) => t.state === state) : [];
  return (
    <div className="instances">
      <div className="row">
        <b className="mono">{state}</b><span className="muted">{doc?.does}</span>
        <button className="linkbtn" style={{ marginLeft: "auto" }} onClick={close}>close</button>
      </div>
      {loops.length + tasks.length === 0 && <div className="empty">No instance is in this state now.</div>}
      {loops.length > 0 && (
        <table className="t">
          <thead><tr><th>{def.instance_of}</th><th>round</th><th className="r">in state</th><th>waiting on</th><th>loop</th></tr></thead>
          <tbody>{loops.map((l) => (
            <tr key={l.id} className="click" onClick={() => nav(def.id === "solve" ? `/cases/${l.subject}` : "/coverage")}>
              <td><b>{l.subject_name}</b> {l.escalated && <Chip tone="human">escalated</Chip>}</td>
              <td className="num">{l.round ? `${l.round}/${l.max_rounds}` : "—"}</td>
              <td className="r num">{dur(since(l.state_since, now))}</td>
              <td>{l.waiting_on ? <span className="mono small">{a.tasks.find((t) => t.id === l.waiting_on)?.type ?? ""} {l.waiting_on.slice(-6)}</span> : "—"}</td>
              <td className="mono small">{l.id.slice(-8)}</td>
            </tr>))}
          </tbody>
        </table>
      )}
      {tasks.length > 0 && (
        <table className="t">
          <thead><tr><th>task</th><th>goal</th><th>slot</th><th className="r">since</th><th>retries</th><th>card</th></tr></thead>
          <tbody>{tasks.map((t) => (
            <tr key={t.id}>
              <td><span className="tag task">{t.type}</span></td><td>{t.goal}{t.last_rejection && <div className="small" style={{ color: "#9b1f1f" }}>last rejection: {t.last_rejection}</div>}</td>
              <td className="mono">{t.slot ?? "—"}</td><td className="r num">{dur(since(t.state_since, now))}</td>
              <td className="num">{t.retries}/{t.max_retries}</td><td><RecordLink id={t.card} /></td>
            </tr>))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function NowPanel({ a }: { a: ActiveRuns }) {
  const now = useNow();
  const loopName = (id: string | null | undefined) => a.loops.find((l) => l.id === id)?.subject_name ?? "";
  const pools = a.workers.flatMap((w) => w.pools.map((p) => ({ ...p, worker: w.name })));
  const lease = a.leases[0];
  const holder = lease && a.jobs.find((j) => j.id === lease.holder_job);
  return (
    <Panel title="Now" sub={`what is occupying the system at ${hhmm(a.as_of)} UTC`} right={<Link to="/running">Running →</Link>}>
      <div className="now-band">
        <div className="now">
          <div className="caption">Agent slots</div>
          {a.agent_slots.map((slot) => {
            const t = a.tasks.find((x) => x.slot === slot && x.state === "running");
            return (
              <div className={`slot ${t ? "busy" : "free"}`} key={slot}>
                <div className="row"><span className="mono small">{slot}</span>
                  {t ? <><span className="tag task">{t.type}</span><span className="small ellipsis">{loopName(t.loop_id)}</span><span className="num small" style={{ marginLeft: "auto" }}>{dur(t.used_wall_s)}</span></>
                    : <span className="muted small">free</span>}
                </div>
                {t && <><div className="small ink2 ellipsis">{t.goal}</div><UsageBar used={t.used_wall_s} limit={t.budget_wall_s} />
                  <div className="small muted">{dur(t.used_wall_s)} of {dur(t.budget_wall_s)} · queued {dur(t.queue_s)} (not charged)</div></>}
              </div>
            );
          })}
        </div>
        <div className="now">
          <div className="caption">GPU exclusive lease</div>
          {lease ? (
            <div className="slot busy">
              <div className="row"><span className="mono small">{lease.gpu}</span><span className="small ellipsis">{holder?.name}</span>
                <span className="num small" style={{ marginLeft: "auto" }}>{lease.since ? dur(since(lease.since, now)) : "—"}</span></div>
              <UsageBar used={lease.since ? since(lease.since, now) : 0} limit={lease.max_s} />
              <div className="small muted">max hold {dur(lease.max_s)}; shared GPU jobs pause while held</div>
            </div>
          ) : <div className="slot free small muted">free</div>}
          <div className="caption">Job pools</div>
          <table className="t compact">
            <tbody>{pools.map((p) => (
              <tr key={p.worker + p.pool}><td className="small">{p.pool} <span className="muted" style={{ fontSize: 10.5 }}>{p.worker.split("-")[0]}</span></td>
                <td className="r small num">{p.running}/{p.capacity}</td>
                <td className="r small num">{p.queued > 0 ? <Chip tone="waiting">{p.queued} queued</Chip> : <span className="muted">—</span>}</td></tr>
            ))}</tbody>
          </table>
        </div>
        <div className="now">
          <div className="caption">Workers</div>
          {a.workers.map((w) => (
            <div key={w.name} className="row" style={{ gap: 6 }}><Chip tone={w.health === "ok" ? "ok" : w.health === "down" ? "fail" : "waiting"}>{w.name}</Chip>
              <span className="small muted">seen {dur(since(w.last_seen, now))} ago</span></div>
          ))}
          <div className="caption">Budget today</div>
          {a.budgets.map((b) => (
            <div key={b.key}>
              <div className="row small"><span>{b.label}</span><span className="num muted" style={{ marginLeft: "auto" }}>{b.used} / {b.limit} {b.unit}</span></div>
              <UsageBar used={b.used} reserved={b.reserved} limit={b.limit} />
            </div>
          ))}
        </div>
        <div className="now">
          <div className="caption">Needs you</div>
          <div className="inbox-callout">
            <span className="badge human">{a.inbox_count}</span>
            <span>items wait for your decision <span className="muted">(Inbox page: WP9b)</span></span>
          </div>
          <div className="caption">Just finished</div>
          {a.recently_finished.map((f) => (
            <Link key={f.id} to={f.link} className="row small" style={{ gap: 6, color: "inherit" }}>
              <Chip tone={f.status === "ok" ? "ok" : "fail"}>{hhmm(f.ended_at)}</Chip><span className="ellipsis">{f.label}</span>
            </Link>
          ))}
        </div>
      </div>
    </Panel>
  );
}

function PipelineStrip() {
  const nav = useNavigate();
  const now = useNow();
  const tr = useApi(() => api.trace(now - 45 * 60_000, now), [Math.floor(now / 60_000)]);
  if (!tr.data) return <Loading error={tr.error} />;
  return (
    <Panel title="Recent pipeline" sub="last 45 minutes, one row per lane — what ran in parallel" right={<Link to="/pipeline">Open pipeline →</Link>}>
      <Swimlanes spans={tr.data.spans} nowMs={now} from={now - 45 * 60_000} to={now} compact onSpanClick={() => nav("/pipeline")} />
    </Panel>
  );
}

