import { useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router";
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

const TAB_SIZE: Record<string, { w: number; h: number }> = { solve: { w: 136, h: 112 }, expand: { w: 136, h: 104 }, task: { w: 128, h: 100 } };
const WRAP: Record<string, Record<string, string>> = {
  solve: { "elk.layered.wrapping.strategy": "MULTI_EDGE", "elk.aspectRatio": "3.2", "elk.layered.spacing.nodeNodeBetweenLayers": "40" },
  expand: { "elk.layered.wrapping.strategy": "MULTI_EDGE", "elk.aspectRatio": "3.2", "elk.layered.spacing.nodeNodeBetweenLayers": "40" },
  task: { "elk.layered.wrapping.strategy": "MULTI_EDGE", "elk.aspectRatio": "2.4", "elk.layered.spacing.nodeNodeBetweenLayers": "34" },
};

function OverviewBody({ m, a }: { m: Machines; a: ActiveRuns }) {
  const [params, setParams] = useSearchParams();
  const tab = params.get("machine") ?? "solve";
  const [sel, setSel] = useState<string>();
  const def = m.machines.find((x) => x.id === tab) ?? m.machines[0];
  const all = useMemo(() => Object.fromEntries(m.machines.map((x) => [x.id, counts(x, a)])), [m, a]);
  const choose = (id: string) => {
    const next = new URLSearchParams(params);
    next.set("machine", id);
    setParams(next, { replace: true });
    setSel(undefined);
  };
  return (
    <div className="stack">
      <Intro a={a} />
      <div className="overview-main">
        <section className="panel primary machine-panel">
          <header className="machine-tabs" role="tablist">
            {m.machines.map((x) => {
              const n = Object.values(all[x.id] ?? {}).reduce((p, q) => p + q, 0);
              return (
                <button key={x.id} role="tab" aria-selected={x.id === def.id} className={x.id === def.id ? "on" : ""} onClick={() => choose(x.id)}>
                  {x.title}<span className="tab-count">{n}</span>
                </button>
              );
            })}
            <span className="tab-hint muted">click a state to list its instances</span>
          </header>
          <div className="body">
            <p className="machine-summary">
              <b>One per {def.instance_of}</b>{def.process !== "any" ? <> · process <span className="mono">{def.process}</span></> : null}
              {def.limits ? <> · {Object.entries(def.limits).map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`).join(" · ")}</> : null} — {def.summary}
            </p>
            <StateMachine key={def.id} machine={def} taskTypes={m.task_types} counts={all[def.id] ?? {}} selected={sel}
              onSelect={setSel} nodeSize={TAB_SIZE[def.id]} options={WRAP[def.id]} minScale={0.8} />
            <AnyState def={def} counts={all[def.id] ?? {}} onSelect={setSel} />
            <DiagramLegend />
            {sel && <Instances def={def} state={sel} a={a} close={() => setSel(undefined)} />}
            <Branches machine={def} taskTypes={m.task_types} />
            {def.notes?.map((n) => <p key={n} className="small muted" style={{ margin: "6px 0 0" }}>Open in design: {n}</p>)}
            <details>
              <summary>Show transitions as a table</summary>
              <TransitionTable def={def} />
            </details>
          </div>
        </section>
        <NowPanel a={a} />
      </div>
      <PipelineStrip />
    </div>
  );
}

/** "From any state" transitions (e.g. you cancel) are drawn as a footnote, not as a floating node. */
function AnyState({ def, counts, onSelect }: { def: MachineDef; counts: Record<string, number>; onSelect: (s: string) => void }) {
  const any = def.transitions.filter((t) => t.from === "*");
  if (!any.length) return null;
  return (
    <div className="any-state">
      {any.map((t) => (
        <button key={t.to} className="linkbtn" onClick={() => onSelect(t.to)}>
          from any state → <span className="mono">{t.to}</span> ({t.when}){counts[t.to] ? ` · ${counts[t.to]} now` : ""}
        </button>
      ))}
    </div>
  );
}

function TransitionTable({ def }: { def: MachineDef }) {
  return (
    <table className="t">
      <thead><tr><th>from</th><th>to</th><th>kind</th><th>when</th><th>creates</th></tr></thead>
      <tbody>
        {def.transitions.map((t, i) => (
          <tr key={i}><td className="mono">{t.from}</td><td className="mono">{t.to}</td><td>{t.rule ? `rule ${t.rule} · ` : ""}{t.kind}</td>
            <td>{t.when ?? ""}</td><td>{t.creates?.join(", ") ?? ""}</td></tr>
        ))}
      </tbody>
    </table>
  );
}

function Intro({ a }: { a: ActiveRuns }) {
  const running = a.tasks.filter((t) => t.state === "running").length;
  const solving = a.loops.filter((l) => l.machine === "solve" && !["accepted", "closed"].includes(l.state)).length;
  const learning = a.loops.filter((l) => l.machine === "expand" && l.state !== "covered").length;
  return (
    <div className="intro">
      <p>
        KTA improves GPU kernels by reusing <b>Templates</b> learned from reference implementations. Deterministic loops drive the work;
        LLM agents only do the creative tasks (violet <span className="tag task">T…</span> labels), and every proposal must pass fixed checks.
      </p>
      <div className="intro-now">
        <span><b className="num">{solving}</b> cases being solved</span>
        <span><b className="num">{learning}</b> references being learned</span>
        <span><b className="num">{running}</b>/{a.agent_slots.length} agent slots busy</span>
        <span><b className="num">{a.jobs.filter((j) => j.state === "running").length}</b> jobs running, <b className="num">{a.jobs.filter((j) => j.state === "queued").length}</b> queued</span>
        <span className="needs-you"><b className="num">{a.inbox_count}</b> need you</span>
      </div>
    </div>
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
  const inbox = useApi(api.inbox, []);
  const loopName = (id: string | null | undefined) => a.loops.find((l) => l.id === id)?.subject_name ?? "";
  const lease = a.leases[0];
  const holder = lease && a.jobs.find((j) => j.id === lease.holder_job);
  return (
    <aside className="panel now-panel">
      <header><h2>Now</h2><span className="sub">{hhmm(a.as_of)} UTC</span><span className="right"><Link to="/running">Running →</Link></span></header>
      <div className="body now">
        <div className="caption">Needs you <Link to="/inbox" style={{ textTransform: "none", fontWeight: 400, marginLeft: 6 }}>Inbox →</Link></div>
        {(inbox.data ?? []).length === 0 ? <div className="small muted">nothing waiting</div> : (inbox.data ?? []).map((i) => (
          <Link key={i.id} to="/inbox" className="inbox-callout" style={{ textDecoration: "none" }}>
            <span className="badge human">!</span>
            <span><b>{i.trigger}</b><br />{i.subject}<br /><span className="muted">{i.suggestion}</span></span>
          </Link>
        ))}

        <div className="caption">Agent slots</div>
        {a.agent_slots.map((slot) => {
          const t = a.tasks.find((x) => x.slot === slot && x.state === "running");
          return (
            <div className={`slot ${t ? "busy" : "free"}`} key={slot}>
              <div className="row"><span className="mono small">{slot}</span>
                {t ? <><span className="tag task">{t.type}</span><span className="small ellipsis">{loopName(t.loop_id)}</span><span className="num small" style={{ marginLeft: "auto" }}>{dur(t.used_wall_s)}</span></>
                  : <span className="muted small">free</span>}
              </div>
              {t && <><div className="small ink2 ellipsis2">{t.goal}</div><UsageBar used={t.used_wall_s} limit={t.budget_wall_s} />
                <div className="small muted">of {dur(t.budget_wall_s)} budget · queued {dur(t.queue_s)}, not charged</div></>}
            </div>
          );
        })}

        <div className="caption">GPU exclusive lease</div>
        {lease ? (
          <div className="slot busy gpu">
            <div className="row"><span className="mono small">{lease.gpu}</span><span className="small ellipsis">{holder?.name}</span>
              <span className="num small" style={{ marginLeft: "auto" }}>{lease.since ? dur(since(lease.since, now)) : "—"}</span></div>
            <UsageBar used={lease.since ? since(lease.since, now) : 0} limit={lease.max_s} />
            <div className="small muted">max {dur(lease.max_s)}; shared GPU jobs wait meanwhile</div>
          </div>
        ) : <div className="slot free small">free</div>}

        <div className="caption">Workers and pools</div>
        {a.workers.map((w) => (
          <div key={w.name} className="worker-line">
            <Chip tone={w.health === "ok" ? "ok" : w.health === "down" ? "fail" : "waiting"}>{w.name}</Chip>
            <span className="small muted">seen {dur(since(w.last_seen, now))} ago</span>
            <div className="pool-pills">{w.pools.map((p) => (
              <span key={p.pool} className={`pool-pill ${p.running ? "busy" : ""}`} title={`${p.pool}: ${p.running} running of ${p.capacity}, ${p.queued} queued`}>
                {p.pool} {p.running}/{p.capacity}{p.queued ? <b> +{p.queued}q</b> : null}
              </span>
            ))}</div>
          </div>
        ))}

        <div className="caption">Budget today</div>
        {a.budgets.map((b) => (
          <div key={b.key}>
            <div className="row small"><span>{b.label}</span><span className="num muted" style={{ marginLeft: "auto" }}>{Math.round(b.used)} / {b.limit} {b.unit}</span></div>
            <UsageBar used={b.used} reserved={b.reserved} limit={b.limit} />
          </div>
        ))}

        {a.recently_finished.length > 0 && <>
          <div className="caption">Just finished</div>
          {a.recently_finished.slice(0, 4).map((f) => (
            <Link key={f.id} to={f.link} className="row small" style={{ gap: 6, color: "inherit" }}>
              <Chip tone={f.status === "ok" ? "ok" : "fail"}>{hhmm(f.ended_at)}</Chip><span className="ellipsis">{f.label}</span>
            </Link>
          ))}
        </>}
      </div>
    </aside>
  );
}

function PipelineStrip() {
  const nav = useNavigate();
  const now = useNow();
  const tr = useApi(() => api.trace(now - 45 * 60_000, now), [Math.floor(now / 60_000)]);
  if (!tr.data) return <Loading error={tr.error} />;
  return (
    <Panel title="Recent pipeline" sub="last 45 minutes, one row per lane — what ran in parallel" right={<Link to="/pipeline">Open pipeline →</Link>}>
      <Swimlanes spans={tr.data.spans} nowMs={now} from={now - 45 * 60_000} to={now} compact rowHeight={17} onSpanClick={() => nav("/pipeline")} />
    </Panel>
  );
}

