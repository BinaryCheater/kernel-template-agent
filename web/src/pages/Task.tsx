import { useMemo, useState } from "react";
import { Link, useParams } from "react-router";
import { api } from "../data/source";
import type { Machines, Span, TaskDetail, TaskRecord, TaskSpan } from "../data/types";
import { useApi } from "../data/useApi";
import { useNow } from "../data/live";
import { TaskActions } from "../components/actions";
import { SpanLegend, Swimlanes, kindStyle } from "../components/Swimlanes";
import { Chip, Loading, Panel, RecordLink, Stat, UsageBar } from "../components/ui";
import { dur, hhmm, stateTone, wordTone } from "../components/format";

export function TaskPage() {
  const { id = "" } = useParams();
  const t = useApi(() => api.task(id), [id]);
  const m = useApi(api.machines, []);
  if (!t.data || !m.data) return <Loading error={t.error ?? m.error} />;
  return <TaskBody d={t.data} m={m.data} />;
}

const toSpan = (s: TaskSpan): Span => ({
  span_id: s.span_id, parent_id: s.parent_id, trace_id: s.trace_id, kind: s.kind, lane: s.lane, name: s.name,
  start_ts: s.start, end_ts: s.end, status: s.status as Span["status"], attrs: s.attrs as Span["attrs"],
});

function TaskBody({ d, m }: { d: TaskDetail; m: Machines }) {
  const now = useNow();
  const t = d.task;
  const def = m.machines.find((x) => x.id === "task");
  const live = !["accepted", "rejected", "cancelled", "escalated"].includes(t.state);
  const typeDef = m.task_types.find((x) => x.id === t.type);
  return (
    <div className="stack">
      <div className="task-head">
        <div>
          <div className="caption">Task · {t.type} {typeDef ? `· ${typeDef.name}` : ""}</div>
          <h2 style={{ fontSize: 19 }}>{d.card.goal}</h2>
          <div className="row small ink2" style={{ marginTop: 4 }}>
            <Chip tone={stateTone(def, t.state)}>{t.state}</Chip>
            {t.slot && <span className="mono">{t.slot}</span>}
            {t.last_rejection && <span style={{ color: "#9b1f1f" }}>last rejection: {t.last_rejection}</span>}
            <span>retries {t.retries}/{t.max_retries}</span>
            <span>priority {t.priority}</span>
            <span>card <RecordLink id={d.card.id} /></span>
            {d.loop && <span>loop {d.loop.machine === "solve" ? <Link to={`/cases/${d.loop.subject}`}>{d.loop.subject_name}</Link> : d.loop.subject_name} · {d.loop.state}</span>}
            <span className="mono muted">{t.id}</span>
          </div>
        </div>
        {live || t.state === "escalated" ? <TaskActions task={t.id} label={`${t.type} ${d.card.goal}`} retry={t.state === "escalated" || t.state === "rejected"} /> : null}
      </div>

      <div className="stats">
        <Stat label="Wall time" value={dur(d.time.wall_s || t.used_wall_s)} foot={`budget ${dur(t.budget_wall_s)}`} />
        <Stat label="In tools and jobs" value={dur(d.time.tool_s)} foot={d.time.wall_s > 0 ? `${Math.round((d.time.tool_s / d.time.wall_s) * 100)}% of wall time` : "—"} />
        <Stat label="Queued" value={dur(d.time.queue_s || t.queue_s)} foot="waiting for a slot or a job" />
        <Stat label="Checks" value={`${d.checks.length - d.rejections.length}/${d.checks.length}`} foot={d.rejections.length ? `${d.rejections.length} failed` : "none failed"} />
      </div>
      <div className="small muted">Time composition from {d.time.source === "AgentRun" ? "AgentRun records" : "closed spans (no AgentRun record yet)"}.</div>

      <Panel primary title="What the agent did" sub="the task's spans: the agent session with its model and tool segments, kta calls and the jobs they started">
        <TaskTimeline spans={d.spans} now={now} />
      </Panel>

      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1.3fr) minmax(0, 1fr)", alignItems: "start" }}>
        <Panel title="kta calls and jobs" sub={`${d.calls.length} calls, in order`}>
          {d.calls.length === 0 ? <div className="empty">No kta call or job span is tagged with this task yet.</div> : (
            <table className="t">
              <thead><tr><th>at</th><th>kind</th><th>call</th><th className="r">took</th><th>status</th></tr></thead>
              <tbody>{d.calls.map((c) => (
                <tr key={c.span_id} className="call-row">
                  <td className="num">{hhmm(c.start, true)}</td>
                  <td><span className="sw-inline" style={{ background: kindStyle(c.kind).color }} />{kindStyle(c.kind).label}</td>
                  <td className="mono">{c.name}<div className="muted">{c.lane}</div></td>
                  <td className="r num">{c.dur_s == null ? "running" : dur(c.dur_s)}</td>
                  <td><Chip tone={wordTone(c.status)}>{c.status}</Chip></td>
                </tr>
              ))}</tbody>
            </table>
          )}
        </Panel>
        <div className="stack">
          <Panel title="Task card" sub="the contract the agent works against">
            <dl className="kv">
              <dt>done checks</dt><dd>{d.card.done_checks.map((c) => <div key={c} className="mono small">{c}</div>)}</dd>
              <dt>inputs</dt><dd className="row wrap" style={{ gap: 6 }}>{d.card.inputs.length ? d.card.inputs.map((i) => <RecordLink key={i} id={i} />) : <span className="muted">none</span>}</dd>
              <dt>budget</dt><dd className="small">{Object.entries(d.card.budget).map(([k, v]) => `${k} ${v}`).join(" · ")}</dd>
              <dt>profile</dt><dd>{d.card.profile && d.card.profile.length > 30 ? <RecordLink id={d.card.profile} /> : <span className="mono">{d.card.profile}</span>}</dd>
              <dt>work unit</dt><dd className="mono small">{d.card.work_unit}</dd>
            </dl>
            {t.budget_wall_s > 0 && <div style={{ marginTop: 8 }}><UsageBar used={t.used_wall_s} limit={t.budget_wall_s} /></div>}
          </Panel>
          <Rejections d={d} />
          {d.human_actions.length > 0 && (
            <Panel title="Your actions on this task">
              {d.human_actions.map((h) => <div key={h.id} className="small"><span className="num">{hhmm(h.at)}</span> <b>{h.action}</b> — {h.reason || <span className="muted">no reason</span>} <RecordLink id={h.id} /></div>)}
            </Panel>
          )}
        </div>
      </div>

      <AgentRuns runs={d.agent_runs} />
      <Panel title="Records written in this task" sub="records whose actor names this task">
        {d.records.length === 0 ? <div className="empty">None yet.</div> : (
          <table className="t">
            <thead><tr><th>at</th><th>kind</th><th>by</th><th>status</th><th>record</th></tr></thead>
            <tbody>{d.records.map((r) => (
              <tr key={r.id}><td className="num">{hhmm(r.created_at)}</td><td>{r.kind}</td><td className="small">{r.actor}</td>
                <td><Chip tone={wordTone(r.status)}>{r.status}</Chip></td><td><RecordLink id={r.id} /></td></tr>
            ))}</tbody>
          </table>
        )}
      </Panel>
    </div>
  );
}

function TaskTimeline({ spans, now }: { spans: TaskSpan[]; now: number }) {
  const all = useMemo(() => spans.map(toSpan), [spans]);
  const expanded = useMemo(() => new Set(all.filter((s) => s.kind === "agent_session").map((s) => s.span_id)), [all]);
  if (all.length === 0) return <div className="empty">No span is tagged with this task yet (the orchestrator writes them when it runs the task).</div>;
  const from = Math.min(...all.map((s) => Date.parse(s.start_ts)));
  const to = Math.max(...all.map((s) => (s.end_ts ? Date.parse(s.end_ts) : now)));
  const pad = Math.max(10_000, (to - from) * 0.03);
  const kinds = [...new Set(all.map((s) => s.kind))];
  return (
    <>
      <Swimlanes spans={all} nowMs={now} from={from - pad} to={to + pad} expanded={expanded} />
      <SpanLegend kinds={kinds} />
    </>
  );
}

function Rejections({ d }: { d: TaskDetail }) {
  if (d.checks.length === 0) return null;
  return (
    <Panel title="Done checks" sub="one verdict per submission, written by the orchestrator's checkers, never by the agent">
      {d.checks.map((c, n) => (
        <div key={c.id} style={{ marginBottom: 8 }}>
          <div className="row small"><b>#{n + 1}</b><Chip tone={c.passed ? "ok" : "fail"}>{c.passed ? "passed" : "rejected"}</Chip>
            <span className="num">{hhmm(c.created_at)}</span><RecordLink id={c.id} /></div>
          {c.checks.map((x) => (
            <div key={x.name} className="small" style={{ paddingLeft: 12, color: x.passed ? undefined : "#9b1f1f" }}>
              {x.passed ? "✓" : "✕"} <span className="mono">{x.name}</span>{x.failures.length ? `: ${x.failures.slice(0, 2).join("; ")}` : ""}
            </div>
          ))}
        </div>
      ))}
    </Panel>
  );
}

function AgentRuns({ runs }: { runs: TaskRecord[] }) {
  const [open, setOpen] = useState<string | null>(null);
  if (runs.length === 0) return (
    <Panel title="Agent runs" sub="AgentRun records (WP8 runner)"><div className="empty">No AgentRun record for this task yet.</div></Panel>
  );
  return (
    <Panel title="Agent runs" sub="one AgentRun per agent step; the prompt is the TASK.md the agent received">
      <table className="t">
        <thead><tr><th>at</th><th>session</th><th>model</th><th className="r">wall</th><th className="r">tools</th><th>exit</th><th>tokens</th><th>violations</th><th /></tr></thead>
        <tbody>{runs.map((r) => {
          const b = r.body as { session: string; model: string; wall_s: number; tool_s: number; exit: string; tokens_in: number | null; tokens_out: number | null; violations: unknown[] };
          return (
            <tr key={r.id}>
              <td className="num">{hhmm(r.created_at)}</td><td className="mono small">{b.session.slice(0, 12)}</td><td className="small">{b.model}</td>
              <td className="r num">{dur(b.wall_s)}</td><td className="r num">{dur(b.tool_s)}</td>
              <td><Chip tone={b.exit === "ok" || b.exit === "submitted" ? "ok" : "waiting"}>{b.exit}</Chip></td>
              <td className="small num">{b.tokens_in != null ? `${b.tokens_in} in / ${b.tokens_out} out` : "—"}</td>
              <td>{b.violations.length ? <Chip tone="fail">{b.violations.length}</Chip> : <span className="muted small">none</span>}</td>
              <td className="row" style={{ gap: 6 }}>{r.blobs.prompt && <button className="linkbtn" onClick={() => setOpen(open === r.blobs.prompt ? null : r.blobs.prompt)}>prompt</button>}<RecordLink id={r.id} /></td>
            </tr>
          );
        })}</tbody>
      </table>
      {open && <BlobView sha={open} />}
    </Panel>
  );
}

function BlobView({ sha }: { sha: string }) {
  const b = useApi(() => api.blob(sha), [sha]);
  if (!b.data) return <Loading error={b.error} />;
  return (
    <div style={{ marginTop: 8 }}>
      <div className="small muted">blob {sha.slice(0, 12)} · {b.data.media} · {b.data.size} bytes{b.data.truncated ? " (truncated)" : ""}</div>
      <pre className="text">{b.data.text}</pre>
    </div>
  );
}
