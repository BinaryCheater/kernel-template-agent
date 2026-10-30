import type { ReactNode } from "react";
import { Link } from "react-router";
import { api, nowMs } from "../data/source";
import type { ActiveRuns, BudgetNode, Job, Machines } from "../data/types";
import { useApi } from "../data/useApi";
import { Chip, Loading, Panel, RecordLink, UsageBar } from "../components/ui";
import { dur, hhmm, since, stateTone } from "../components/format";
import { kindStyle } from "../components/Swimlanes";

export function Running() {
  const a = useApi(api.active, []);
  const m = useApi(api.machines, []);
  if (!a.data || !m.data) return <Loading error={a.error ?? m.error} />;
  return <RunningBody a={a.data} m={m.data} />;
}

function RunningBody({ a, m }: { a: ActiveRuns; m: Machines }) {
  const now = nowMs();
  const loopName = (id?: string) => a.loops.find((l) => l.id === id)?.subject_name ?? "";
  const jobsOf = (pool: string, worker: string) => a.jobs.filter((j) => j.pool === pool && j.worker === worker);
  return (
    <div className="stack">
      <Panel primary title="Resource board" sub="every slot the scheduler can hand out; dashed = free capacity">
        <div className="board">
          <BoardRow label="Agent slots" hint="LLM worker processes (default 2)">
            {a.agent_slots.map((slot) => {
              const t = a.tasks.find((x) => x.slot === slot && x.state === "running");
              return t ? (
                <div className="cell busy agent" key={slot}>
                  <div className="row"><span className="mono small">{slot}</span><span className="tag task">{t.type}</span>
                    <span className="small" style={{ marginLeft: "auto" }}>{dur(t.used_wall_s)}</span></div>
                  <div className="cell-title">{loopName(t.loop_id)}</div>
                  <div className="small ink2 ellipsis2">{t.goal}</div>
                  <UsageBar used={t.used_wall_s} limit={t.budget_wall_s} />
                  <div className="small muted">budget {dur(t.budget_wall_s)} · card <RecordLink id={t.card} /></div>
                </div>
              ) : <div className="cell free" key={slot}><span className="mono small">{slot}</span> free</div>;
            })}
          </BoardRow>
          {a.workers.flatMap((w) => w.pools.map((p) => {
            const jobs = jobsOf(p.pool, w.name);
            const running = jobs.filter((j) => j.state === "running");
            const queued = jobs.filter((j) => j.state === "queued");
            const lease = p.pool === "gpu-exclusive" ? a.leases.find((l) => l.gpu.startsWith(w.name)) : undefined;
            const freeCells = Math.min(p.capacity - running.length, 4);
            return (
              <BoardRow key={w.name + p.pool} label={`${p.pool}`} hint={`${w.name} · ${p.running}/${p.capacity} running${queued.length ? ` · ${queued.length} queued` : ""}`}>
                {running.map((j) => <JobCell key={j.id} j={j} now={now} lease={lease} owner={loopName(j.loop_id)} />)}
                {Array.from({ length: freeCells }, (_, i) => <div key={i} className="cell free">free</div>)}
                {p.capacity - running.length > 4 && <div className="cell more muted small">+{p.capacity - running.length - 4} more free</div>}
                {queued.length > 0 && (
                  <div className="cell queued">
                    <Chip tone="waiting">{queued.length} queued</Chip>
                    {queued.map((j) => <div key={j.id} className="small ellipsis">{j.name}{j.pool === "gpu-shared" && a.leases.length ? " — paused by exclusive lease" : ""}</div>)}
                  </div>
                )}
              </BoardRow>
            );
          }))}
        </div>
      </Panel>

      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1.4fr) minmax(0, 1fr)", alignItems: "start" }}>
        <Panel title="Active loop instances" sub="one row per Case or Reference being worked on">
          <table className="t">
            <thead><tr><th>loop</th><th>subject</th><th>state</th><th>round</th><th className="r">in state</th><th>waiting on</th></tr></thead>
            <tbody>{a.loops.map((l) => {
              const def = m.machines.find((x) => x.id === l.machine);
              const wt = a.tasks.find((t) => t.id === l.waiting_on);
              return (
                <tr key={l.id}>
                  <td className="small">{l.machine}</td>
                  <td>{l.machine === "solve" ? <Link to={`/cases/${l.subject}`}>{l.subject_name}</Link> : l.subject_name}
                    {l.escalated && <> <Chip tone="human">escalated</Chip></>}</td>
                  <td><Chip tone={stateTone(def, l.state)}>{l.state}</Chip></td>
                  <td className="num">{l.round ? `${l.round}/${l.max_rounds}` : "—"}</td>
                  <td className="r num">{dur(since(l.state_since, now))}</td>
                  <td className="small">{wt ? <><span className="tag task">{wt.type}</span> {wt.state}</> : "—"}</td>
                </tr>
              );
            })}</tbody>
          </table>
        </Panel>
        <Panel title="Budgets" sub="used · reserved (light) · limit; reservations are taken before a task starts">
          <div className="stack" style={{ gap: 12 }}>{a.budgets.map((b) => <BudgetTree key={b.key} b={b} />)}</div>
        </Panel>
      </div>

      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1.4fr) minmax(0, 1fr)", alignItems: "start" }}>
        <Panel title="Tasks not finished" sub="queued tasks wait for a slot and a budget reservation">
          <table className="t">
            <thead><tr><th>type</th><th>goal</th><th>state</th><th>prio</th><th>retries</th><th className="r">queued</th><th>card</th></tr></thead>
            <tbody>{a.tasks.filter((t) => t.state !== "accepted").map((t) => (
              <tr key={t.id}>
                <td><span className="tag task">{t.type}</span></td>
                <td>{t.goal}<div className="small muted">{loopName(t.loop_id)}</div>
                  {t.last_rejection && <div className="small" style={{ color: "#9b1f1f" }}>last rejection: {t.last_rejection}</div>}</td>
                <td><Chip tone={stateTone(m.machines.find((x) => x.id === "task"), t.state)}>{t.state}</Chip></td>
                <td className="num">{t.priority}</td>
                <td className="num">{t.retries}/{t.max_retries}</td>
                <td className="r num">{dur(t.queue_s)}</td>
                <td><RecordLink id={t.card} /></td>
              </tr>
            ))}</tbody>
          </table>
        </Panel>
        <Panel title="Workers" sub="selected by capability labels, not host names">
          <table className="t">
            <thead><tr><th>worker</th><th>labels</th><th>commit</th><th className="r">last seen</th></tr></thead>
            <tbody>{a.workers.map((w) => (
              <tr key={w.name}>
                <td><Chip tone={w.health === "ok" ? "ok" : w.health === "down" ? "fail" : "waiting"}>{w.name}</Chip>
                  <div className="small muted">{w.host} · {w.transport} · <RecordLink id={w.record} /></div></td>
                <td className="small">{Object.entries(w.labels).map(([k, v]) => <div key={k}><span className="muted">{k}=</span>{v}</div>)}</td>
                <td className="mono small">{w.commit}</td>
                <td className="r num small">{dur(since(w.last_seen, now))} ago</td>
              </tr>
            ))}</tbody>
          </table>
        </Panel>
      </div>

      <Panel title="Finished recently">
        <table className="t">
          <tbody>{a.recently_finished.map((f) => (
            <tr key={f.id}><td style={{ width: 70 }} className="num">{hhmm(f.ended_at)}</td>
              <td><Chip tone={f.status === "ok" ? "ok" : "fail"}>{f.status === "ok" ? "done" : "failed"}</Chip></td>
              <td><Link to={f.link}>{f.label}</Link></td></tr>
          ))}</tbody>
        </table>
      </Panel>
    </div>
  );
}

function BoardRow({ label, hint, children }: { label: string; hint: string; children: ReactNode }) {
  return (
    <div className="board-row">
      <div className="board-label"><b>{label}</b><div className="small muted">{hint}</div></div>
      <div className="board-cells">{children}</div>
    </div>
  );
}

function JobCell({ j, now, lease, owner }: { j: Job; now: number; lease?: { since: string; max_s: number }; owner: string }) {
  const st = kindStyle(j.kind);
  const run = j.started_at ? since(j.started_at, now) : 0;
  return (
    <div className="cell busy" style={{ borderLeftColor: st.color }}>
      <div className="row"><span className="small" style={{ color: st.color, fontWeight: 600 }}>{j.kind}</span>
        <span className="small num" style={{ marginLeft: "auto" }}>{dur(run)}</span></div>
      <div className="cell-title ellipsis">{j.name}</div>
      <div className="small muted">{owner || "agent tool call"} · queued {dur(since(j.queued_at, now) - run)}</div>
      {lease && <><UsageBar used={since(lease.since, now)} limit={lease.max_s} /><div className="small muted">exclusive lease, max {dur(lease.max_s)}</div></>}
    </div>
  );
}

function BudgetTree({ b, depth = 0 }: { b: BudgetNode; depth?: number }) {
  return (
    <div style={{ paddingLeft: depth * 16 }}>
      <div className="row small" style={{ gap: 8 }}>
        <span style={{ fontWeight: depth === 0 ? 600 : 400 }}>{b.label}</span>
        <span className="muted mono" style={{ fontSize: 10.5 }}>{b.key}</span>
        <span className="num muted" style={{ marginLeft: "auto" }}>{b.used} used · {b.reserved} reserved / {b.limit} {b.unit}</span>
      </div>
      <UsageBar used={b.used} reserved={b.reserved} limit={b.limit} />
      {b.used > b.limit && b.limit > 0 && <div className="small" style={{ color: "#9b1f1f" }}>over quota — this process starts no new tasks</div>}
      {b.children?.map((c) => <div key={c.key} style={{ marginTop: 6 }}><BudgetTree b={c} depth={depth + 1} /></div>)}
    </div>
  );
}
