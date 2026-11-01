// Home (WP25, D665): what the system is doing now and what it just did. Top: one status line
// (is anything moving, is anything stopped, what needs you). Middle: the solve and expand
// loops drawn as flows with every live instance placed on its state; each state lists the
// conditions that send work elsewhere, so the logic is read off the picture. Bottom: recent
// activity in plain words, what needs you, the machine's resources and the pipeline strip.
import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router";
import { api } from "../data/source";
import type { ActiveRuns, LoopInstance, MachineDef, Machines, TransitionDef } from "../data/types";
import { useApi } from "../data/useApi";
import { useNow } from "../data/live";
import { Chip, Loading, Panel, UsageBar } from "../components/ui";
import { dur, hhmm, since } from "../components/format";
import { Swimlanes } from "../components/Swimlanes";
import { LoopControls, loopCondition } from "../components/LoopControls";
import { JournalList } from "./Activity";

export function Overview() {
  const machines = useApi(api.machines, []);
  const active = useApi(api.active, []);
  if (!machines.data || !active.data) return <Loading error={machines.error ?? active.error} />;
  return <Home m={machines.data} a={active.data} />;
}

function Home({ m, a }: { m: Machines; a: ActiveRuns }) {
  const solve = m.machines.find((x) => x.id === "solve");
  const expand = m.machines.find((x) => x.id === "expand");
  const task = m.machines.find((x) => x.id === "task");
  return (
    <div className="stack">
      <StatusLine a={a} />
      {solve && <LoopMap def={solve} loops={a.loops.filter((l) => l.machine === "solve")} a={a} m={m} />}
      {expand && <LoopMap def={expand} loops={a.loops.filter((l) => l.machine === "expand")} a={a} m={m} />}
      {task && <TaskStrip def={task} a={a} />}
      <div className="home-bottom">
        <Panel title="What just happened" sub="milestones in plain words, newest first" right={<Link to="/activity">Full activity log →</Link>}>
          <JournalList level="milestone" limit={14} compact />
        </Panel>
        <div className="stack">
          <NeedsYou a={a} />
          <Resources a={a} />
        </div>
      </div>
      <PipelineStrip />
    </div>
  );
}

// ---------------------------------------------------------------- status line

function StatusLine({ a }: { a: ActiveRuns }) {
  const now = useNow();
  const release = useApi(api.release, []);
  const last = useApi(() => api.journal({ limit: 1, level: "all" }), []);
  const live = a.loops.filter((l) => !["accepted", "closed", "covered"].includes(l.state));
  const blocked = live.filter((l) => l.blocked);
  const paused = live.filter((l) => !l.blocked && l.paused);
  const moving = live.filter((l) => !l.blocked && !l.paused);
  const agents = a.tasks.filter((t) => t.state === "running").length;
  const jobs = a.jobs.filter((j) => j.state === "running").length;
  const queued = a.jobs.filter((j) => j.state === "queued").length;
  const lastAt = last.data?.entries[0]?.at;
  const overall = blocked.length
    ? { tone: "fail" as const, text: `${blocked.length} loop${blocked.length > 1 ? "s" : ""} stopped by an error` }
    : moving.length
      ? { tone: "running" as const, text: `${moving.length} loop${moving.length > 1 ? "s" : ""} moving` }
      : paused.length
        ? { tone: "idle" as const, text: "everything paused" }
        : { tone: "idle" as const, text: "idle: no loop is open" };
  return (
    <div className={`status-line ${overall.tone}`}>
      <span className="status-main"><span className="dot" />{overall.text}</span>
      <span>stage <b>{release.data?.stage.id ?? "…"}</b> {release.data?.stage.name}</span>
      <span><b className="num">{a.loops.filter((l) => l.machine === "solve" && !["accepted", "closed"].includes(l.state)).length}</b> cases open</span>
      <span><b className="num">{a.loops.filter((l) => l.machine === "expand" && l.state !== "covered").length}</b> references open</span>
      <span>agents <b className="num">{agents}</b>/{a.agent_slots.length}</span>
      <span>jobs <b className="num">{jobs}</b> running{queued ? <>, <b className="num">{queued}</b> queued</> : null}</span>
      <span>GPU {a.leases.length ? <b>measuring</b> : "free"}</span>
      <span>{a.workers.map((w) => <Chip key={w.name} tone={w.health === "ok" ? "ok" : w.health === "down" ? "fail" : "waiting"}>{w.name}</Chip>)}</span>
      <span className="muted">last event {lastAt ? `${dur(since(lastAt, now))} ago` : "—"}</span>
      <span className="spacer" />
      {a.inbox_count > 0 && <Link to="/inbox" className="needs-link">{a.inbox_count} need{a.inbox_count === 1 ? "s" : ""} you →</Link>}
    </div>
  );
}

// ---------------------------------------------------------------- loop maps

/** The main path: from the initial state follow the first transition that moves forward,
 *  skipping terminals and side waits (a wait state left by `resume` and never entered by a
 *  normal transition, such as solve's `waiting`). Everything else is drawn below the path. */
function mainPath(def: MachineDef): string[] {
  const kind = Object.fromEntries(def.states.map((s) => [s.id, s.kind]));
  const side = new Set(def.states.filter((s) => s.kind === "wait"
    && def.transitions.some((t) => t.from === s.id && t.kind === "resume")
    && !def.transitions.some((t) => t.to === s.id && t.kind === "normal")).map((s) => s.id));
  const path: string[] = [];
  let cur = def.states.find((s) => s.initial)?.id ?? def.states[0].id;
  while (cur && !path.includes(cur)) {
    path.push(cur);
    const next = def.transitions.find((t) => t.from === cur && (t.kind === "normal" || t.kind === "creates_task")
      && kind[t.to] !== "terminal" && !side.has(t.to) && !path.includes(t.to));
    cur = next?.to ?? "";
  }
  return path;
}

const RULE = ["", "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧"];

function exitText(t: TransitionDef): string {
  const head = t.rule ? `${RULE[t.rule] ?? t.rule} ` : "";
  return `${head}${t.label ?? t.when ?? ""}`;
}

function LoopMap({ def, loops, a, m }: { def: MachineDef; loops: LoopInstance[]; a: ActiveRuns; m: Machines }) {
  const path = useMemo(() => mainPath(def), [def]);
  const [sel, setSel] = useState<string | null>(null);
  const below = def.states.filter((s) => !path.includes(s.id));
  const at = (state: string) => loops.filter((l) => l.state === state);
  const focus = sel ?? (loops.find((l) => l.blocked)?.state ?? null);
  return (
    <section className="panel primary loop-map">
      <header>
        <h2>{def.title}</h2>
        <span className="sub">one per {def.instance_of.toLowerCase()} · {def.summary}</span>
        <span className="right">{loops.length} open · click a state to see its rules</span>
      </header>
      <div className="body">
        <div className="lm-row" style={{ gridTemplateColumns: `repeat(${path.length}, minmax(0, 1fr))` }}>
          {path.map((id, i) => {
            const s = def.states.find((x) => x.id === id)!;
            const fwd = i < path.length - 1 ? def.transitions.find((t) => t.from === id && t.to === path[i + 1]) : undefined;
            const exits = def.transitions.filter((t) => t.from === id && t.to !== path[i + 1] && t.kind !== "any");
            return (
              <div key={id} className="lm-cell">
                <StateNode def={def} state={id} kind={s.kind} work={s.work} does={s.does} loops={at(id)} on={focus === id} onClick={() => setSel(focus === id ? null : id)} />
                {fwd && <div className="lm-arrow" title={fwd.when ?? ""} />}
                {(exits.length > 0 || fwd?.label) && (
                  <ul className="lm-exits">
                    {fwd?.label && <li className="fwd" title={fwd.when ?? ""}><span className="to">→ {fwd.to}</span> {fwd.label}{fwd.creates && <span className="tag task">{fwd.creates.join("/")}</span>}</li>}
                    {exits.map((t) => (
                      <li key={`${t.to}-${t.rule ?? t.label}`} title={t.when ?? ""}>
                        <span className="to">→ {t.to}</span> {exitText(t)}{t.creates && <span className="tag task">{t.creates.join("/")}</span>}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            );
          })}
        </div>
        <div className="lm-below">
          {below.map((s) => {
            const into = def.transitions.filter((t) => t.to === s.id && t.from !== "*");
            const out = def.transitions.filter((t) => t.from === s.id);
            return (
              <div key={s.id} className={`lm-side ${s.kind}`}>
                <StateNode def={def} state={s.id} kind={s.kind} work={s.work} does={s.does} loops={at(s.id)} on={focus === s.id} onClick={() => setSel(focus === s.id ? null : s.id)} />
                <div className="small ink2">
                  <div><span className="muted">entered from </span>{[...new Set(into.map((t) => t.from))].join(", ") || "—"}</div>
                  {out.length > 0 && <div><span className="muted">then </span>{out.map((t) => `${t.to}${t.label ? ` (${t.label})` : ""}`).join(" · ")}</div>}
                </div>
              </div>
            );
          })}
        </div>
        {focus && <StateDetail def={def} state={focus} loops={at(focus)} a={a} m={m} close={() => setSel(null)} />}
      </div>
    </section>
  );
}

function StateNode(p: { def: MachineDef; state: string; kind: string; work: string; does: string; loops: LoopInstance[]; on: boolean; onClick: () => void }) {
  const nav = useNavigate();
  const now = useNow();
  const bad = p.loops.some((l) => l.blocked);
  return (
    <div className={`lm-node ${p.kind} ${p.loops.length ? "has" : ""} ${bad ? "bad" : ""} ${p.on ? "on" : ""}`} onClick={p.onClick} role="button">
      <div className="lm-name"><b className="mono">{p.state}</b><span className="lm-work">{p.work}</span></div>
      <div className="lm-does">{p.does}</div>
      {p.loops.length > 0 && (
        <div className="lm-chips">
          {p.loops.map((l) => {
            const c = loopCondition(l);
            return (
              <span key={l.id} className={`lm-chip ${c.tone}`} title={`${l.subject_name}: ${c.text} · in ${l.state} for ${dur(since(l.state_since, now))}`}
                onClick={(e) => { e.stopPropagation(); nav(p.def.id === "solve" ? `/cases/${l.subject}` : `/activity?case=${l.subject}`); }}>
                {l.subject_name}{l.round ? ` R${l.round}` : ""} · {c.tone === "running" ? dur(since(l.state_since, now)) : c.text}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}

function StateDetail({ def, state, loops, a, m, close }: { def: MachineDef; state: string; loops: LoopInstance[]; a: ActiveRuns; m: Machines; close: () => void }) {
  const now = useNow();
  const s = def.states.find((x) => x.id === state);
  const exits = def.transitions.filter((t) => t.from === state || (t.from === "*" && t.to !== state));
  const types = Object.fromEntries(m.task_types.map((t) => [t.id, t.name]));
  return (
    <div className="lm-detail">
      <div className="row"><b className="mono">{state}</b><span className="ink2">{s?.does}</span><span className="muted small">done by {s?.work}</span>
        <button className="linkbtn" style={{ marginLeft: "auto" }} onClick={close}>close</button></div>
      <div className="lm-detail-grid">
        <div>
          <div className="caption">Leaves this state when</div>
          <ul className="rules">
            {exits.map((t, i) => (
              <li key={i}><span className="mono">→ {t.to}</span> {t.rule ? <b>{RULE[t.rule]}</b> : null} {t.when ?? t.label ?? "the step's work is done"}
                {t.creates?.map((c) => <span key={c} className="tag task" title={types[c]}>{c} {types[c]}</span>)}
                {t.kind === "resume" && <span className="muted"> (when the task is accepted)</span>}</li>
            ))}
            {exits.length === 0 && <li className="muted">terminal: the loop ends here</li>}
          </ul>
          {exits.some((t) => t.rule) && <div className="small muted">①–⑥ are checked in this order; the first that holds decides.</div>}
        </div>
        <div>
          <div className="caption">Here now</div>
          {loops.length === 0 && <div className="small muted">nothing is in this state</div>}
          {loops.map((l) => {
            const c = loopCondition(l);
            const task = l.waiting_on ? a.tasks.find((t) => t.id === l.waiting_on) : undefined;
            return (
              <div key={l.id} className="lm-inst">
                <div className="row">
                  {def.id === "solve" ? <Link to={`/cases/${l.subject}`}><b>{l.subject_name}</b></Link> : <b>{l.subject_name}</b>}
                  <span className="small ink2">round {l.round ?? "—"}{l.max_rounds ? `/${l.max_rounds}` : ""} · in {state} {dur(since(l.state_since, now))}</span>
                  <Chip tone={c.tone}>{c.text}</Chip>
                </div>
                {l.blocked && <div className="small error">{l.blocked.error}: {l.blocked.message} <span className="muted">(since {hhmm(l.blocked.at)} UTC)</span></div>}
                {task && <div className="small ink2">waiting on <span className="tag task">{task.type}</span> {task.goal} · {task.state}</div>}
                <LoopControls loop={l} />
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function TaskStrip({ def, a }: { def: MachineDef; a: ActiveRuns }) {
  const now = useNow();
  if (a.tasks.length === 0) {
    return <div className="task-strip muted small">Agent tasks: none queued or running. A loop creates one (T1–T6) when it cannot go on by itself; the violet labels above show where.</div>;
  }
  return (
    <Panel title="Agent tasks" sub="tasks the loops created, by lifecycle state">
      <div className="task-states">
        {def.states.filter((s) => a.tasks.some((t) => t.state === s.id)).map((s) => (
          <div key={s.id} className="task-state">
            <div className="caption">{s.id}</div>
            {a.tasks.filter((t) => t.state === s.id).map((t) => (
              <Link key={t.id} to={`/tasks/${t.id}`} className="task-pill">
                <span className="tag task">{t.type}</span> <span className="ellipsis">{t.goal}</span>
                <span className="small muted">{dur(since(t.state_since, now))}{t.slot ? ` · ${t.slot}` : ""}</span>
              </Link>
            ))}
          </div>
        ))}
      </div>
    </Panel>
  );
}

// ---------------------------------------------------------------- right column

function NeedsYou({ a }: { a: ActiveRuns }) {
  const inbox = useApi(api.inbox, []);
  const items = inbox.data ?? [];
  return (
    <Panel title="Needs you" right={<Link to="/inbox">Inbox →</Link>}>
      {items.length === 0 && <div className="small muted">Nothing waits for you.</div>}
      {items.map((i) => {
        const loop = i.id.startsWith("blocked:") ? a.loops.find((l) => l.id === i.id.slice(8)) : undefined;
        return (
          <div key={i.id} className="needs-item">
            <div><b>{i.trigger}</b></div>
            <div className="small">{loop ? <Link to={`/cases/${loop.subject}`}>{loop.subject_name}</Link> : null} {i.subject}</div>
            <div className="small muted">{i.suggestion}</div>
            {loop ? <LoopControls loop={loop} /> : i.task ? <Link className="small" to={`/tasks/${i.task}`}>open the task →</Link> : null}
          </div>
        );
      })}
    </Panel>
  );
}

function Resources({ a }: { a: ActiveRuns }) {
  const now = useNow();
  const lease = a.leases[0];
  return (
    <Panel title="Machine" right={<Link to="/running">Details →</Link>}>
      <div className="res-grid">
        {a.agent_slots.map((slot) => {
          const t = a.tasks.find((x) => x.slot === slot && x.state === "running");
          return (
            <div key={slot} className={`slot ${t ? "busy" : "free"}`}>
              <div className="row"><span className="mono small">{slot}</span>{t ? <span className="tag task">{t.type}</span> : <span className="muted small">free</span>}</div>
              {t && <UsageBar used={t.used_wall_s} limit={t.budget_wall_s} />}
            </div>
          );
        })}
        <div className={`slot ${lease ? "busy gpu" : "free"}`}>
          <div className="row"><span className="mono small">GPU exclusive</span><span className="small muted">{lease?.since ? dur(since(lease.since, now)) : "free"}</span></div>
        </div>
      </div>
      {a.workers.map((w) => (
        <div key={w.name} className="pool-pills" style={{ marginTop: 6 }}>
          {w.pools.map((p) => <span key={p.pool} className={`pool-pill ${p.running ? "busy" : ""}`}>{w.name} {p.pool} {p.running}/{p.capacity}{p.queued ? <b> +{p.queued}q</b> : null}</span>)}
        </div>
      ))}
    </Panel>
  );
}

function PipelineStrip() {
  const nav = useNavigate();
  const now = useNow();
  const tr = useApi(() => api.trace(now - 45 * 60_000, now), [Math.floor(now / 60_000)]);
  if (!tr.data) return <Loading error={tr.error} />;
  return (
    <Panel title="Recent pipeline" sub="last 45 minutes, one row per lane" right={<Link to="/pipeline">Open pipeline →</Link>}>
      <Swimlanes spans={tr.data.spans} nowMs={now} from={now - 45 * 60_000} to={now} compact rowHeight={17} onSpanClick={() => nav("/pipeline")} />
    </Panel>
  );
}
